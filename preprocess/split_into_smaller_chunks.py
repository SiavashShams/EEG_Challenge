"""Split .npy > 10 minutes into smaller chunks to speedup loading"""
import glob
import json
import os
import pickle

import numpy as np

import re

EEG_SR = 64
WAV_SR = 16000

CHUNK_SEC = 10
HOP_SEC = 5

CHUNK_EEG = CHUNK_SEC * EEG_SR
HOP_EEG = HOP_SEC * EEG_SR
CHUNK_WAV = CHUNK_SEC * WAV_SR
HOP_WAV = HOP_SEC * WAV_SR

if __name__ == '__main__':

    in_folder = '/engram/naplab/shared/eeg_challenge_data_new/derivatives/split_data_16khz'
    out_folder = '/engram/naplab/shared/eeg_challenge_data_new/derivatives/split_data_16khz_10sec'
    os.makedirs(out_folder, exist_ok=True)

    all_eeg_npy = glob.glob(os.path.join(in_folder, '*eeg.npy'))
    all_wav_npy = glob.glob(os.path.join(in_folder, '*wav.npy'))

    for eeg_npy in all_eeg_npy:
        in_name = os.path.basename(eeg_npy)
        long_eeg = np.load(eeg_npy)
        length = len(long_eeg)
        duration = length / EEG_SR
        n_chunk = length // HOP_EEG - 1 # The last chunk is longer to avoid zero-padding
        print(f'Spliting {str(in_name)} of {str(round(duration))} seconds into {str(n_chunk)} chunks...')

        for i in range(n_chunk):
            if i == n_chunk - 1:
                chunk_eeg = long_eeg[i*HOP_EEG:] # to the end
                out_name = in_name.replace('eeg.npy', f'eeg_{str(i*HOP_SEC)}-{str(round(duration))}sec.npy')
            else:
                chunk_eeg = long_eeg[i*HOP_EEG:i*HOP_EEG+CHUNK_EEG]
                out_name = in_name.replace('eeg.npy', f'eeg_{str(i*HOP_SEC)}-{str(i*HOP_SEC+CHUNK_SEC)}sec.npy')
            np.save(os.path.join(out_folder, out_name), chunk_eeg)

    for wav_npy in all_wav_npy:
        in_name = os.path.basename(wav_npy)
        long_wav = np.load(wav_npy)
        length = len(long_wav)
        duration = length / WAV_SR
        n_chunk = length // HOP_WAV - 1 # The last chunk is longer to avoid zero-padding
        print(f'Spliting {str(in_name)} of {str(round(duration))} seconds into {str(n_chunk)} chunks...')

        for i in range(n_chunk):
            if i == n_chunk - 1:
                chunk_wav = long_wav[i*HOP_WAV:] # to the end
                out_name = in_name.replace('wav.npy', f'wav_{str(i*HOP_SEC)}-{str(round(duration))}sec.npy')
            else:
                chunk_wav = long_wav[i*HOP_WAV:i*HOP_WAV+CHUNK_WAV]
                out_name = in_name.replace('wav.npy', f'wav_{str(i*HOP_SEC)}-{str(i*HOP_SEC+CHUNK_SEC)}sec.npy')
            np.save(os.path.join(out_folder, out_name), chunk_wav)