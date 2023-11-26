'''
TCN Implemented by 
https://github.com/exporl/auditory-eeg-challenge-2024-code
Translated to pytorch by ChatGPT
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwinDilationModel(nn.Module):
    def __init__(
        self,
        eeg_input_dimension=64,
        sti_input_dimension=1,
        layers=3,
        kernel_size=3,
        spatial_filters=8,
        dilation_filters=16,
        activation=nn.ReLU(),
        num_mismatched_segments=4,
        
    ):
        super(TwinDilationModel, self).__init__()

        self.eeg_input_dimension = eeg_input_dimension
        self.sti_input_dimension = sti_input_dimension
        self.layers = layers
        self.kernel_size = kernel_size
        self.spatial_filters = spatial_filters
        self.dilation_filters = dilation_filters
        self.activation = activation
        self.num_mismatched_segments = num_mismatched_segments

        self.eeg_input = nn.Conv1d(in_channels=eeg_input_dimension, out_channels=spatial_filters, kernel_size=1)
        self.sti_input = nn.Conv1d(in_channels=sti_input_dimension, out_channels=spatial_filters, kernel_size=1)

        self.eeg_tcn = nn.Sequential()
        self.sti_tcn = nn.Sequential()

        for layer_index in range(layers):
            # Dilation on EEG
            self.eeg_tcn.append(nn.Conv1d(
                in_channels=dilation_filters if layer_index else spatial_filters,
                out_channels=dilation_filters,
                kernel_size=kernel_size,
                dilation=kernel_size**layer_index,
                stride=1,
                padding=0,
            ))
            self.eeg_tcn.append(activation)

            # Dilation on stimuli
            self.sti_tcn.append(nn.Conv1d(
                in_channels=dilation_filters if layer_index else spatial_filters,
                out_channels=dilation_filters,
                kernel_size=kernel_size,
                dilation=kernel_size**layer_index,
                stride=1,
                padding=0,
            ))
            self.sti_tcn.append(activation)

        self.linear_proj_sim = nn.Linear(dilation_filters*dilation_filters, 1)

    def forward(self, eeg, sti):
        '''
        eeg: (B, 64, T)
        sti: (B, M, 1, T)
        '''
        B, M, C, T = sti.shape
        sti = sti.view(B*M, C, T)

        eeg_proj = self.eeg_input(eeg) # (B, 8, T)
        sti_proj = self.sti_input(sti) # (B*M, 8, T)
        
        eeg_proj = self.eeg_tcn(eeg_proj) # (B, 16, T')
        sti_proj = self.sti_tcn(sti_proj) # (B*M, 16, T')

        sti_proj = sti_proj.view(B, M, *sti_proj.shape[-2:]) # (B, M, 16, T')

        # Normalize in Time
        eeg_proj_norm = F.normalize(eeg_proj, p=2, dim=-1)
        sti_proj_norm = F.normalize(sti_proj, p=2, dim=-1)

        # Pairwise cosine similarity
        cos = torch.stack([
            torch.bmm(
                eeg_proj_norm, sti_proj_norm[:, i].permute(0, 2, 1)
            ).view(B, -1) 
            for i in range(5)
            ],dim=1 
        ) # (B, M, 16*16)

        # Linear projection of similarity matrices
        pred = self.linear_proj_sim(cos).squeeze(-1) # (B, M)

        return pred
