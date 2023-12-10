# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import logging
import math
import typing as tp

import torch
from torch import nn

logger = logging.getLogger(__name__)


def pad_multiple(x: torch.Tensor, base: int):
    length = x.shape[-1]
    target = math.ceil(length / base) * base
    return torch.nn.functional.pad(x, (0, target - length))


class ScaledEmbedding(nn.Module):
    """Scale up learning rate for the embedding, otherwise, it can move too slowly.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, scale: float = 10.):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self):
        return self.embedding.weight * self.scale

    def forward(self, x):
        return self.embedding(x) * self.scale


class SubjectLayers(nn.Module):
    """Per subject linear layer."""
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, init_id: bool = False):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels))
        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= 1 / in_channels**0.5

    def forward(self, x, subjects):
        _, C, D = self.weights.shape
        weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
        return torch.einsum("bct,bcd->bdt", x, weights)

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """
    def __init__(self, channels: int, init: float = 0.1, boost: float = 5.):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init / boost
        self.boost = boost

    def forward(self, x):
        return (self.boost * self.scale[:, None]) * x


class ConvSequence(nn.Module):

    def __init__(self, channels: tp.Sequence[int], kernel: int = 4, dilation_growth: int = 1,
                 dilation_period: tp.Optional[int] = None, stride: int = 2,
                 dropout: float = 0.0, leakiness: float = 0.0, groups: int = 1,
                 decode: bool = False, batch_norm: bool = False, dropout_input: float = 0,
                 skip: bool = False, scale: tp.Optional[float] = None, rewrite: bool = False,
                 activation_on_last: bool = True, post_skip: bool = False, glu: int = 0,
                 glu_context: int = 0, glu_glu: bool = True, activation: tp.Any = None) -> None:
        super().__init__()
        
        dilation = 1
        channels = tuple(channels)
        print("channels:",channels)
        self.skip = skip
        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()
        if activation is None:
            activation = partial(nn.LeakyReLU, leakiness)
        Conv = nn.Conv1d if not decode else nn.ConvTranspose1d
        # build layers
        for k, (chin, chout) in enumerate(zip(channels[:-1], channels[1:])):
            layers: tp.List[nn.Module] = []
            is_last = k == len(channels) - 2

            # Set dropout for the input of the conv sequence if defined
            if k == 0 and dropout_input:
                assert 0 < dropout_input < 1
                layers.append(nn.Dropout(dropout_input))

            # conv layer
            if dilation_growth > 1:
                assert kernel % 2 != 0, "Supports only odd kernel with dilation for now"
            if dilation_period and (k % dilation_period) == 0:
                dilation = 1
            pad = kernel // 2 * dilation
            layers.append(Conv(chin, chout, kernel, stride, pad,
                               dilation=dilation, groups=groups if k > 0 else 1))
            dilation *= dilation_growth
            # non-linearity
            if activation_on_last or not is_last:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(num_features=chout))
                layers.append(activation())
                if dropout:
                    layers.append(nn.Dropout(dropout))
                if rewrite:
                    layers += [nn.Conv1d(chout, chout, 1), nn.LeakyReLU(leakiness)]
                    # layers += [nn.Conv1d(chout, 2 * chout, 1), nn.GLU(dim=1)]
            if chin == chout and skip:
                if scale is not None:
                    layers.append(LayerScale(chout, scale))
                if post_skip:
                    layers.append(Conv(chout, chout, 1, groups=chout, bias=False))

            self.sequence.append(nn.Sequential(*layers))
            if glu and (k + 1) % glu == 0:
                ch = 2 * chout if glu_glu else chout
                act = nn.GLU(dim=1) if glu_glu else activation()
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(chout, ch, 1 + 2 * glu_context, padding=glu_context), act))
            else:
                self.glus.append(None)

    def forward(self, x: tp.Any) -> tp.Any:
        for module_idx, module in enumerate(self.sequence):
            old_x = x
            x = module(x)
            if self.skip and x.shape == old_x.shape:
                x = x + old_x
            glu = self.glus[module_idx]
            if glu is not None:
                x = glu(x)
        return x


class DualPathRNN(nn.Module):
    def __init__(self, channels: int, depth: int, inner_length: int = 10):
        super().__init__()
        self.lstms = nn.ModuleList([nn.LSTM(channels, channels, 1) for _ in range(depth * 4)])
        self.inner_length = inner_length

    def forward(self, x: torch.Tensor):
        B, C, L = x.shape
        IL = self.inner_length
        x = pad_multiple(x, self.inner_length)
        x = x.permute(2, 0, 1).contiguous()
        for idx, lstm in enumerate(self.lstms):
            y = x.reshape(-1, IL, B, C)
            if idx % 2 == 0:
                y = y.transpose(0, 1).reshape(IL, -1, C)
            else:
                y = y.reshape(-1, IL * B, C)
            y, _ = lstm(x)
            if idx % 2 == 0:
                y = y.reshape(IL, -1, B, C).transpose(0, 1).reshape(-1, B, C)
            else:
                y = y.reshape(-1, B, C)
            x = x + y

            if idx % 2 == 1:
                x = x.flip(dims=(0,))
        return x[:L].permute(1, 2, 0).contiguous()
