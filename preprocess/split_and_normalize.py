"""Split data in sets and normalize (per recording)."""
import glob
import json
import os
import pickle


import numpy as np

import re

EEG_SR = 64
WAV_SR = 16000
ENV_SR = 64

if __name__ == "__main__":

    # Arguments for splitting and normalizing
    speech_features = ['wav', 'env']
    splits = [90, 10] # [80, 10, 10]
    split_names = ['train', 'valid']
    overwrite = True

    # Calculate the split fraction
    split_fractions = [x/sum(splits) for x in splits]

    # Construct the necessary paths
    processed_eeg_folder = '/engram/naplab/shared/eeg_challenge_data_new/derivatives/preprocessed_eeg'
    processed_stimuli_folder = '/engram/naplab/shared/eeg_challenge_data_new/derivatives/preprocessed_stimuli_16khz'
    split_data_folder = '/engram/naplab/shared/eeg_challenge_data_new/derivatives/split_data_16khz'

    # Create the output folder
    os.makedirs(split_data_folder, exist_ok=True)

    # Find all subjects
    all_subjects = glob.glob(os.path.join(processed_eeg_folder, "sub*"))
    nb_subjects = len(all_subjects)
    print(f"Found {nb_subjects} subjects to split/normalize")

    # Loop over subjects
    for subject_index, subject_path in enumerate(all_subjects):
        subject = os.path.basename(subject_path)
        print(f"Starting with subject {subject} ({subject_index + 1}/{nb_subjects})...")
        # Find all recordings
        all_recordings = glob.glob(os.path.join(subject_path, "*", "*.npy"))
        print(f"\tFound {len(all_recordings)} recordings for subject {subject}.")
        # Loop over recordings
        for recording_index, recording in enumerate(all_recordings):
            print(f"\tStarting with recording {recording} ({recording_index + 1}/{len(all_recordings)})...")

            # Load EEG from disk
            print(f"\t\tLoading EEG for {recording}")

            eeg = np.load(recording)

            # swap axes to have time as first dimension
            eeg = np.swapaxes(eeg, 0, 1)

            # keep only the 64 channels
            eeg = eeg[:, :64]

            # retrieve the stimulus name from the filename
            stimulus_filename = recording.split('_eeg.')[0].split('-audio-')[1]

            # Load wav and env from disk
            print(f"\t\tLoading wav and env for recording {recording} ")
            wav_path = os.path.join(
                processed_stimuli_folder,
                stimulus_filename + "_-_" + 'wav' + ".npy",
            )
            wav = np.load(wav_path)
            env_path = os.path.join(
                processed_stimuli_folder,
                stimulus_filename + "_-_" + 'envelope' + ".npy",
            )
            env = np.load(env_path)

            len_ratio = wav.shape[0]/eeg.shape[0]
            print('wav.shape[0]/eeg.shape[0]:', round(len_ratio, 2))
            assert abs(len_ratio - WAV_SR/EEG_SR) < 50 # audiobook_2_2
            len_ratio = env.shape[0]/eeg.shape[0]
            print('env.shape[0]/eeg.shape[0]:', round(len_ratio, 2))

            # Do the actual splitting
            print(f"\t\tSplitting/normalizing recording {recording}...")

            # Left-aligned or center-aligned ???
            eeg_sec = eeg.shape[0] / EEG_SR
            wav_sec = wav.shape[0] / WAV_SR
            env_sec = env.shape[0] / ENV_SR
            min_sec = min(eeg_sec, wav_sec, env_sec)
            print('min_sec: ', min_sec)

            start_sec = 0
            for split_name, split_fraction in zip(split_names, split_fractions): 
                end_sec = min(start_sec+split_fraction*min_sec, min_sec)
                cut_eeg = eeg[int(start_sec*EEG_SR):int(end_sec*EEG_SR), ...]
                cut_wav = wav[int(start_sec*WAV_SR):int(end_sec*WAV_SR), ...]
                cut_env = env[int(start_sec*ENV_SR):int(end_sec*ENV_SR), ...]

                print(f'({str(start_sec)}, {str(end_sec)})')

                # Save
                eeg_filename = f"{split_name}_-_{subject}_-_{stimulus_filename}_-_eeg.npy"
                eeg_save_path = os.path.join(split_data_folder, eeg_filename)
                if not os.path.exists(eeg_save_path) or overwrite:
                    np.save(eeg_save_path, cut_eeg)
                else:
                    print(f"\t\tSkipping {eeg_filename} because it already exists")

                wav_filename = f"{split_name}_-_{subject}_-_{stimulus_filename}_-_wav.npy"
                wav_save_path = os.path.join(split_data_folder, wav_filename)
                if not os.path.exists(wav_save_path) or overwrite:
                    np.save(wav_save_path, cut_wav)
                else:
                    print(f"\t\tSkipping {wav_filename} because it already exists")

                env_filename = f"{split_name}_-_{subject}_-_{stimulus_filename}_-_envelope.npy"
                env_save_path = os.path.join(split_data_folder, env_filename)
                if not os.path.exists(env_save_path) or overwrite:
                    np.save(env_save_path, cut_env)
                else:c
                    print(f"\t\tSkipping {env_filename} because it already exists")

                start_sec = end_sec
