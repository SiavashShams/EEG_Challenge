'''
TCN Implemented by 
https://github.com/exporl/auditory-eeg-challenge-2024-code
Translated to pytorch by ChatGPT
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class TCN(nn.Module):

    def __init__(
        self,
        input_dim=64,
        layers=3,
        kernel_size=3,
        spatial_filters=8,
        dilation_filters=16,
        activation=nn.ReLU()
        
    ):
        super(TCN, self).__init__()

        self.input_dim = input_dim
        self.layers = layers
        self.kernel_size = kernel_size
        self.spatial_filters = spatial_filters
        self.dilation_filters = dilation_filters
        self.activation = activation

        self.conv1x1 = nn.Conv1d(in_channels=input_dim, out_channels=spatial_filters, kernel_size=1)
        self.tcn = nn.Sequential()

        for layer_index in range(layers):
            # Dilation on EEG
            self.tcn.append(nn.Conv1d(
                in_channels=dilation_filters if layer_index else spatial_filters,
                out_channels=dilation_filters,
                kernel_size=kernel_size,
                dilation=kernel_size**layer_index,
                stride=1,
                padding=0,
            ))
            self.tcn.append(activation)


    def forward(self, x):
        '''
        x: (B, C, T) or (B, M, C, T)
        '''
        n_dim = x.dim()
        if n_dim == 4: # Batch of windows
            B, M, C, T = x.shape
            x = x.view(B*M, C, T)

        h = self.conv1x1(x)
        h = self.tcn(h)

        if n_dim == 4:
            h = h.view(B, M, *h.shape[-2:])

        return h


class ProjSimilarityClassifier(nn.Module):

    def __init__(
        self,
        input_dim=16
    ):
        super(ProjSimilarityClassifier, self).__init__()
        self.linear_proj_sim = nn.Linear(input_dim*input_dim, 1)

    def forward(self, h1, h2):
        '''
        h1: target EEG embedding of shape (B, C, T) 
        h2: M candidates speech embedding of shape (B, M, C, T)
        '''
        B, M, _, _ = h2.shape

        # Normalize in Time
        h1_norm = F.normalize(h1, p=2, dim=-1)
        h2_norm = F.normalize(h2, p=2, dim=-1)

        # Pairwise cosine similarity
        cos = torch.stack([
            torch.bmm(
                h1_norm, h2_norm[:, i].permute(0, 2, 1)
            ).view(B, -1) 
            for i in range(M)
            ],dim=1 
        ) # (B, M, 16*16)

        pred = self.linear_proj_sim(cos).squeeze(-1) # (B, M)

        return pred


class TwinDilationModel(nn.Module):

    def __init__(
        self,
        eeg_input_dimension=64,
        sti_input_dimension=1,
        layers=3,
        kernel_size=3,
        spatial_filters=8,
        dilation_filters=16,
        activation=nn.ReLU()
        
    ):
        super(TwinDilationModel, self).__init__()

        self.eeg_input_dimension = eeg_input_dimension
        self.sti_input_dimension = sti_input_dimension
        self.layers = layers
        self.kernel_size = kernel_size
        self.spatial_filters = spatial_filters
        self.dilation_filters = dilation_filters
        self.activation = activation

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
            for i in range(sti_proj_norm.shape[1])
            ],dim=1 
        ) # (B, M, 16*16)

        # Linear projection of similarity matrices
        pred = self.linear_proj_sim(cos).squeeze(-1) # (B, M)

        return pred
