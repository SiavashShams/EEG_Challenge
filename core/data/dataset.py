import os
import glob
import random
import itertools

import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import resample


class RandomWindowDataset(Dataset):
    
    def __init__(
        self, 
        data_root,
        split,
        speech_features=['envelope'],
        eeg_sr=64, 
        feature_sr=[64], 
        n_win=5,
        win_sec=5,
        rand_win=True,
    ):
        if not isinstance(speech_features, list):
            speech_features = [speech_features]
        if not isinstance(feature_sr, list):
            feature_sr = [feature_sr]
        self.features = ['eeg'] + speech_features
        self.rates = {fname: sr for fname, sr in zip(speech_features, feature_sr)}
        self.rates['eeg'] = eeg_sr

        files = [
            x for x in glob.glob(os.path.join(data_root, f"{split}_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] 
            in self.features
        ]
        self.files = self.group_recordings(files)

        self.n_win, self.win_sec = n_win, win_sec
        self.rand_win = rand_win

    def group_recordings(self, files):
        new_files = []
        grouped = itertools.groupby(sorted(files), lambda x: "_-_".join(os.path.basename(x).split("_-_")[:3]))
        for recording_name, feature_paths in grouped:
            subject_files = {}
            for path in feature_paths:
                for fname in self.features:
                    if fname in os.path.basename(path):
                        subject_files[fname] = path
                        break
            new_files.append(subject_files)

        return new_files
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        subject_files = self.files[idx]

        # Load all required features
        subject_npy = {
            fname: np.load(subject_files[fname], mmap_mode='r').astype(np.float32)
            for fname in self.features
        }
        duration = min([len(feature)/self.rates[fname] for fname, feature in subject_npy.items()])

        # Randomly choose windows
        # IMPORTANT: time is the FIRST dim
        windows = {fname: [] for fname in self.features}
        for i in range(self.n_win):
            if self.rand_win:
                start_sec = random.uniform(0, duration-self.win_sec)
            else:
                start_sec = (duration-self.win_sec) / (self.n_win+1) * (i+1)

            for fname in self.features:
                start = int(start_sec*self.rates[fname])
                end = start + int(self.win_sec*self.rates[fname])
                window = subject_npy[fname][start:end]
                if window.ndim == 1:
                    window = window[:, np.newaxis]
                # window shape (win_len, dim)
                windows[fname].append(window)

        # Collect all windows into array
        for fname in self.features:
            windows[fname] = np.array(windows[fname]).transpose(0, 2, 1) # (n_win, dim, win_len)

        return windows

# IMPORTANT: time is the FIRST dim
def extract_windows(signal, window_size, stride):
    assert len(signal) > 1000
    signal_length = len(signal)
    num_windows = (signal_length - window_size) // stride + 1

    # Calculate the starting indices for each window
    start_indices = np.arange(0, num_windows * stride, stride)

    # Use array slicing to extract windows
    windows = signal[start_indices[:, np.newaxis] + np.arange(window_size)]

    return windows


class AllWindowDataset(Dataset):
    
    def __init__(
        self, 
        data_root,
        split,
        speech_features=['envelope'],
        eeg_sr=64, 
        feature_sr=[64], 
        n_win=5,
        win_sec=5,
        hop_sec=1
    ):
        if not isinstance(speech_features, list):
            speech_features = [speech_features]
        if not isinstance(feature_sr, list):
            feature_sr = [feature_sr]
        self.features = ['eeg'] + speech_features
        self.rates = {fname: sr for fname, sr in zip(speech_features, feature_sr)}
        self.rates['eeg'] = eeg_sr

        files = [
            x for x in glob.glob(os.path.join(data_root, f"{split}_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] 
            in self.features
        ]
        self.files = self.group_recordings(files)
        self.n_win, self.win_sec, self.hop_sec = n_win, win_sec, hop_sec
        self.windows = self.concatenate_all_windows(self.files)
    
    def concatenate_all_windows(self, files):
        self.windows = {fname: [] for fname in self.features}
        for file_dict in files:
            for fname, path in file_dict.items():
                # IMPORTANT: time is the FIRST dim
                feature = np.load(path, mmap_mode='r').astype(np.float32)
                if feature.ndim == 1:
                    feature = feature[:, np.newaxis]
                windows = extract_windows(
                    feature, 
                    window_size=self.win_sec*self.rates[fname],
                    stride=self.hop_sec*self.rates[fname]
                )
                print(fname, windows.shape)
                self.windows[fname].append(windows)
        
        for fname in self.features:
            self.windows[fname] = np.array(self.windows[fname])


    def group_recordings(self, files):
        new_files = []
        grouped = itertools.groupby(sorted(files), lambda x: "_-_".join(os.path.basename(x).split("_-_")[:3]))
        for recording_name, feature_paths in grouped:
            subject_files = {}
            for path in feature_paths:
                for fname in self.features:
                    if fname in os.path.basename(path):
                        subject_files[fname] = path
                        break
            new_files.append(subject_files)

        return new_files
    
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return {fname: self.windows[fname][idx] for fname in self.features}


# Depreciated
class RandomWindowDatasetSplit(Dataset):
    
    def __init__(
        self, 
        data_root,
        split,
        eeg_sr=64, 
        wav_sr=16000, 
        n_win=5,
        win_sec=5,
        rand_win=True,
        use_env=False,
        env_extractor=None,
        down_env=False
    ):
        eeg_files = [x for x in glob.glob(os.path.join(data_root, f"{split}_-_*")) 
            if os.path.basename(x).split("_-_")[-1].split(".")[0].split("_")[0] in ['eeg']
        ]
        self.eeg_files = self.group_recordings(eeg_files)

        self.eeg_sr, self.wav_sr = eeg_sr, wav_sr
        self.n_win, self.win_sec = n_win, win_sec
        self.rand_win = rand_win
        self.use_env, self.env_extractor = use_env, env_extractor
        self.down_env = down_env

    def group_recordings(self, files):
        new_files = []
        grouped = itertools.groupby(sorted(files), lambda x: "_-_".join(os.path.basename(x).split("_-_")[:3]))
        for recording_name, feature_paths in grouped:
            new_files += [sorted(feature_paths, key=lambda x: "0" if x == "eeg" else x)]
        return new_files

    def __len__(self):
        return len(self.eeg_files)

    def __getitem__(self, idx):
        subject_files = self.eeg_files[idx]
        subject_eeg_files = random.choices(subject_files, k=self.n_win)
        subject_wav_files = [f.replace('_eeg', '_wav') for f in subject_eeg_files]

        eeg_windows = []
        wav_windows = []

        for i in range(self.n_win):
            eeg = np.load(subject_eeg_files[i], mmap_mode='r').astype(np.float32)
            wav = np.load(subject_wav_files[i], mmap_mode='r').astype(np.float32)
            duration = min(len(eeg)/self.eeg_sr, len(wav)/self.wav_sr)
            start_sec = random.uniform(0, duration-self.win_sec)
            eeg_start = int(start_sec*self.eeg_sr)
            eeg_end = eeg_start + int(self.win_sec*self.eeg_sr)
            wav_start = int(start_sec*self.wav_sr)
            wav_end = wav_start + int(self.win_sec*self.wav_sr)

            eeg_window = eeg[eeg_start:eeg_end]
            wav_window = wav[wav_start:wav_end]

            if self.use_env:
                wav_window = self.env_extractor(wav_window)
                if self.down_env:
                    wav_window = resample(x=wav_window, num=eeg_window.shape[0], axis=0)

            if wav_window.ndim == 1:
                wav_window = wav_window[:, np.newaxis]

            eeg_windows.append(eeg_window)
            wav_windows.append(wav_window)

        eeg_windows = np.array(eeg_windows).transpose(0, 2, 1) # (n_win, 64, eeg_win)
        wav_windows = np.array(wav_windows).transpose(0, 2, 1) # (n_win, 1, wav_win)
 
        return eeg_windows, wav_windows