# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import random
import typing as tp

import torch
from torch import nn
from torch.nn import functional as F
import torchaudio as ta

from .common import (
    ConvSequence, ScaledEmbedding, SubjectLayers,
    DualPathRNN, pad_multiple
)


class SimpleConv(nn.Module):
    def __init__(self,
                 # Channels
                 in_channels: tp.Dict[str, int],
                 out_channels: int,
                 hidden: tp.Dict[str, int],
                 # Overall structure
                 depth: int = 4,
                 concatenate: bool = False,  # concatenate the inputs
                 linear_out: bool = False,
                 complex_out: bool = False,
                 # Conv layer
                 kernel_size: int = 5,
                 growth: float = 1.,
                 dilation_growth: int = 2,
                 dilation_period: tp.Optional[int] = None,
                 skip: bool = False,
                 post_skip: bool = False,
                 scale: tp.Optional[float] = None,
                 rewrite: bool = False,
                 groups: int = 1,
                 glu: int = 0,
                 glu_context: int = 0,
                 glu_glu: bool = True,
                 gelu: bool = False,
                 # Dual path RNN
                 dual_path: int = 0,
                 # Dropouts, BN, activations
                 conv_dropout: float = 0.0,
                 dropout_input: float = 0.0,
                 batch_norm: bool = False,
                 relu_leakiness: float = 0.0,
                 # Subject specific settings
                 n_subjects: int = 200,
                 subject_dim: int = 64,
                 subject_layers: bool = False,
                 subject_layers_dim: str = "input",  # or hidden
                 subject_layers_id: bool = False,
                 embedding_scale: float = 1.0,
                 # stft transform
                 n_fft: tp.Optional[int] = None,
                 fft_complex: bool = True,
                 # Attention multi-dataset support
                 merger: bool = False,
                 merger_pos_dim: int = 256,
                 merger_channels: int = 270,
                 merger_dropout: float = 0.2,
                 merger_penalty: float = 0.,
                 merger_per_subject: bool = False,
                 dropout: float = 0.,
                 dropout_rescale: bool = True,
                 initial_linear: int = 0,
                 initial_depth: int = 1,
                 initial_nonlin: bool = False,
                 subsample_meg_channels: int = 0,
                 ):
        super().__init__()
        if set(in_channels.keys()) != set(hidden.keys()):
            raise ValueError("Channels and hidden keys must match "
                             f"({set(in_channels.keys())} and {set(hidden.keys())})")
        self._concatenate = concatenate
        self.out_channels = out_channels
        print("in_channels",in_channels)
        if gelu:
            activation = nn.GELU
        elif relu_leakiness:
            activation = partial(nn.LeakyReLU, relu_leakiness)
        else:
            activation = nn.ReLU

        assert kernel_size % 2 == 1, "For padding to work, this must be verified"

        self.merger = None
        self.dropout = None
        self.subsampled_meg_channels: tp.Optional[list] = None
        if subsample_meg_channels:
            assert 'meg' in in_channels
            indexes = list(range(in_channels['meg']))
            rng = random.Random(1234)
            rng.shuffle(indexes)
            self.subsampled_meg_channels = indexes[:subsample_meg_channels]

        self.initial_linear = None
        if dropout > 0.:
            self.dropout = ChannelDropout(dropout, dropout_rescale)
        if merger:
            self.merger = ChannelMerger(
                merger_channels, pos_dim=merger_pos_dim, dropout=merger_dropout,
                usage_penalty=merger_penalty, n_subjects=n_subjects, per_subject=merger_per_subject)
            in_channels["meg"] = merger_channels

        if initial_linear:
            print("initial_linear",initial_linear)
            init = [nn.Conv1d(in_channels["meg"], initial_linear, 1)]
            for _ in range(initial_depth - 1):
                init += [activation(), nn.Conv1d(initial_linear, initial_linear, 1)]
            if initial_nonlin:
                init += [activation()]
            self.initial_linear = nn.Sequential(*init)
            print(in_channels["meg"].shape)
            in_channels["meg"] = initial_linear

        self.subject_layers = None
        if subject_layers:
            assert "meg" in in_channels
            meg_dim = in_channels["meg"]
            dim = {"hidden": hidden["meg"], "input": meg_dim}[subject_layers_dim]
            self.subject_layers = SubjectLayers(meg_dim, dim, n_subjects, subject_layers_id)
            in_channels["meg"] = dim

        self.stft = None
        if n_fft is not None:
            assert "meg" in in_channels
            self.fft_complex = fft_complex
            self.n_fft = n_fft
            self.stft = ta.transforms.Spectrogram(
                n_fft=n_fft,
                hop_length=n_fft//2,
                normalized=True,
                power=None if fft_complex else 1,
                return_complex=True)
            in_channels["meg"] *= n_fft // 2 + 1
            if fft_complex:
                in_channels["meg"] *= 2

        self.subject_embedding = None
        if subject_dim:
            self.subject_embedding = ScaledEmbedding(n_subjects, subject_dim, embedding_scale)
            in_channels["meg"] += subject_dim
            print("subject_dim")

        # concatenate inputs if need be
        if concatenate:
            in_channels = {"concat": sum(in_channels.values())}
            hidden = {"concat": sum(hidden.values())}
            print("concat,in_channels",in_channels)
        # compute the sequences of channel sizes
        sizes = {}
        for name in in_channels:
            sizes[name] = [in_channels[name]]
            sizes[name] += [int(round(hidden[name] * growth ** k)) for k in range(depth)]
            print("in_channels", in_channels)
        params: tp.Dict[str, tp.Any]
        params = dict(kernel=kernel_size, stride=1,
                      leakiness=relu_leakiness, dropout=conv_dropout, dropout_input=dropout_input,
                      batch_norm=batch_norm, dilation_growth=dilation_growth, groups=groups,
                      dilation_period=dilation_period, skip=skip, post_skip=post_skip, scale=scale,
                      rewrite=rewrite, glu=glu, glu_context=glu_context, glu_glu=glu_glu,
                      activation=activation)

        final_channels = sum([x[-1] for x in sizes.values()])
        self.dual_path = None
        if dual_path:
            self.dual_path = DualPathRNN(final_channels, dual_path)
        self.final = None
        pad = 0
        kernel = 1
        stride = 1
        if n_fft is not None:
            pad = n_fft // 4
            kernel = n_fft
            stride = n_fft // 2

        if linear_out:
            assert not complex_out
            self.final = nn.ConvTranspose1d(final_channels, out_channels, kernel, stride, pad)
        elif complex_out:
            self.final = nn.Sequential(
                nn.Conv1d(final_channels, 2 * final_channels, 1),
                activation(),
                nn.ConvTranspose1d(2 * final_channels, out_channels, kernel, stride, pad))
        else:
            assert len(sizes) == 1, "if no linear_out, there must be a single branch."
            params['activation_on_last'] = False
            list(sizes.values())[0][-1] = out_channels
        print("sizes:", sizes)
        self.encoders = nn.ModuleDict({name: ConvSequence(channels, **params)
                                       for name, channels in sizes.items()})

    def forward(self, x):
        #subjects = batch.subject_index
            n_dim = x.dim()
            T = x.shape[-1]  # Define T based on the last dimension of x

            if n_dim == 4:  # Batch of windows
                B, M, C, T = x.shape
                x = x.view(B * M, C, T)

            # Assuming 'meg' key is used for input 'x'
            if self.subsampled_meg_channels is not None:
                mask = torch.zeros_like(x[:1, :, :1])
                mask[:, self.subsampled_meg_channels] = 1.
                x = x * mask

            if self.dropout is not None:
                x = self.dropout(x)

            if self.initial_linear is not None:
                x = self.initial_linear(x)

            # Process with STFT if applicable
            if self.stft is not None:
                pad = self.n_fft // 4
                x = F.pad(x, (pad, pad), mode='reflect')
                x = self.stft(x)
                B, C, Fr, T = x.shape
                if self.fft_complex:
                    x = torch.view_as_real(x).permute(0, 1, 2, 4, 3).reshape(B, -1, T)

            # Run through encoders
            x = self.encoders['meg'](x)

            # Dual path and final layers if applicable
            if self.dual_path is not None:
                x = self.dual_path(x)
            if self.final is not None:
                x = self.final(x)

            # Adjust shape for output
            if n_dim == 4:
                x = x.view(B, M, *x.shape[-2:])
            assert x.shape[-1] >= T
            return x[:, :, :T]
