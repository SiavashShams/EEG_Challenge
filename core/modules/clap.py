'''
Copied and modified from 
https://github.com/microsoft/CLAP/blob/main/msclap/models/clap.py
'''


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class EncoderProjector(nn.Module):
    def __init__(self, base, d_in: int, d_out: int) -> None:
        super().__init__()
        self.base = base
        self.projection = Projection(d_in, d_out)

    def forward(self, x):
        embed = self.base(x)
        nd = embed.ndim
        if nd == 3:
            embed = embed.permute(0, 2, 1).contiguous()
        proj = self.projection(embed)
        if nd == 3:
            proj = proj.permute(0, 2, 1).contiguous()
        return proj


class CLAP(nn.Module):

    def __init__(self,
        speech_model,
        eeg_model,
        d_speech: int,
        d_eeg: int,
        d_proj: int,
        tau=0.07
    ):
        super().__init__()

        self.speech_encoder = EncoderProjector(
            base=speech_model,
            d_in=d_speech,
            d_out=d_proj
        )

        self.eeg_encoder = EncoderProjector(
            base=eeg_model,
            d_in=d_eeg,
            d_out=d_proj
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tau))

    def forward(self, speech, eeg):
        speech_proj = self.speech_encoder(speech) # (M, C, T) or (M, D)
        eeg_proj = self.eeg_encoder(eeg) # (M, C, T) or (M, D)

        return speech_proj, eeg_proj, self.logit_scale.exp()

    def compute_similarity_with_scale(self, emb1, emb2, scale):

        emb1 = torch.flatten(emb1, 1) # (B, D)
        emb2 = torch.flatten(emb2, 1) # (B, D)

        emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
        emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)
        sim = scale * emb1 @ emb2.T
        # clamp the scaled logit to avoid instability
        sim = torch.clamp(sim, max=100)
        
        return sim

    def predict_speech(self, speech_proj, eeg_proj):
        '''
        speech_proj: (B, M, D) or (B, M, C, T)
        eeg_proj: (B, D) or (B, C, T)
        '''

        speech_proj = torch.flatten(speech_proj, 2) # (B, M, D=CT)
        eeg_proj = torch.flatten(eeg_proj, 1) # (B, D=CT)


        # Normalize features
        speech_proj_norm = speech_proj / speech_proj.norm(dim=-1, keepdim=True)
        eeg_proj_norm = eeg_proj / eeg_proj.norm(dim=-1, keepdim=True)

        cos = torch.bmm(speech_proj_norm, eeg_proj_norm.unsqueeze(-1)).squeeze(-1) # (B, M)
        pred = torch.argmax(cos, dim=-1) # (B)

        return pred

