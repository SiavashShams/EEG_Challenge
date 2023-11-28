'''
Modified from
https://github.com/exporl/auditory-eeg-challenge-2024-code/blob/main/util/dataset_generator.py
'''

import os
import glob
import random
import itertools
import numpy as np

import torch
import tensorflow as tf
from torch.utils.data import Dataset

from tqdm import tqdm


def shuffle_fn(args, number_mismatch):
    # repeat the last argument number_mismatch times
    args = list(args)
    for _  in range(number_mismatch):
        args.append(tf.random.shuffle(args[-1]))
    return tuple(args)


def shuffle_signals_N_times(args, N):
    assert len(args) == 2
    eeg, wav = args
    eeg_list = [eeg]
    wav_list = [wav]

    for n in range(N):
        indices = tf.range(start=0, limit=tf.shape(eeg)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        peeg = tf.gather(eeg, shuffled_indices)
        pwav = tf.gather(wav, shuffled_indices)

        eeg_list.append(peeg)
        wav_list.append(pwav)

    eeg_list = tf.stack(eeg_list, axis=1)
    wav_list = tf.stack(wav_list, axis=1)

    return (eeg_list, wav_list)


class DataGenerator:
    """Generate data for the Match/Mismatch task."""

    def __init__(
        self,
        data_root,
        split,
        features=['eeg', 'wav'],
        tf=True
    ):
        """Initialize the DataGenerator.

        Parameters
        ----------
        files: Sequence[Union[str, pathlib.Path]]
            Files to load.
        """
        files = [x for x in glob.glob(os.path.join(data_root, f"{split}_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        self.files = self.group_recordings(files)
        np.random.shuffle(self.files)
        self.tf = tf

    def group_recordings(self, files):
        """Group recordings and corresponding stimuli.

        Parameters
        ----------
        files : Sequence[Union[str, pathlib.Path]]
            List of filepaths to preprocessed and split EEG and speech features

        Returns
        -------
        list
            Files grouped by the self.group_key_fn and subsequently sorted
            by the self.feature_sort_fn.
        """
        new_files = []
        grouped = itertools.groupby(sorted(files), lambda x: "_-_".join(os.path.basename(x).split("_-_")[:3]))
        for recording_name, feature_paths in grouped:
            new_files += [sorted(feature_paths, key=lambda x: "0" if x == "eeg" else x)]
        return new_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, recording_index):
        """Get data for a certain recording.

        Parameters
        ----------
        recording_index: int
            Index of the recording in this dataset

        Returns
        -------
        Union[Tuple[tf.Tensor,...], Tuple[np.ndarray,...]]
            The features corresponding to the recording_index recording
        """
        data = []
        for feature in self.files[recording_index]:
            f = np.load(feature).astype(np.float32)
            if f.ndim == 1:
                f = f[:,None]

            data += [f]
        data = self.prepare_data(data)
        if self.tf:
            return tuple(tf.constant(x) for x in data)
        else:
            return tuple(x for x in data)


    def __call__(self):
        """Load data for the next recording.

        Yields
        -------
        Union[Tuple[tf.Tensor,...], Tuple[np.ndarray,...]]
            The features corresponding to the recording_index recording
        """
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)

            if idx == self.__len__() - 1:
                self.on_epoch_end()

    def on_epoch_end(self):
        """Change state at the end of an epoch."""
        np.random.shuffle(self.files)

    def prepare_data(self, data):
        return data
    
    
def create_tf_dataset(
    data_root,
    split,
    win_sec=5,
    hop_sec=1,
    features=['eeg', 'wav'],
    feature_dims=(64, 1),
    feature_rates=(64, 16000),
    number_mismatch=4, # None for regression, 2 or 4 for match-mismatch
    batch_size=64,
    n_worker=tf.data.AUTOTUNE
):
    """Creates a tf.data.Dataset.

    This will be used to create a dataset generator that will
    pass windowed data to a model in both tasks.

    Parameters
    ---------
    data_generator: DataGenerator
        A data generator.
    batch_equalizer_fn: Callable
        Function that will be applied on the data after batching (using
        the `map` method from tf.data.Dataset). In the match/mismatch task,
        this function creates the imposter segments and labels.
    batch_size: Optional[int]
        If not None, specifies the batch size. In the match/mismatch task,
        this amount will be doubled by the default_batch_equalizer_fn
    data_types: Union[Sequence[tf.dtype], tf.dtype]
        The data types that the individual features of data_generator should
        be cast to. If you only specify a single datatype, it will be chosen
        for all EEG/speech features.

    Returns
    -------
    tf.data.Dataset
        A Dataset object that generates data to train/evaluate models
        efficiently
    """
    # create tf dataset from generator
    data_generator = DataGenerator(
        data_root=data_root,
        split=split,
        features=features,
        tf=True
    )
    
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=tuple(
            tf.TensorSpec(shape=(None, x), dtype=tf.float32)
            for index, x in enumerate(feature_dims)
        ),
    ) # ( (Neeg, 64), (Nwav, 1) )

    # window dataset
    dataset = dataset.map(
        lambda *args: [
            tf.signal.frame(arg, int(win_sec*feature_rates[i]), int(hop_sec*feature_rates[i]), axis=0)
            for i, arg in enumerate(args)
        ],
        num_parallel_calls=n_worker
    ) # (n_win, win_len, 64 or 1)

    if number_mismatch is not None:
        # map second argument to shifted version
        dataset = dataset.map( lambda *args : shuffle_signals_N_times(args, number_mismatch),
            num_parallel_calls=n_worker
        ) # ( eeg, wav0, wav1(shuffled), wav2(shuffled), ... )

    dataset = dataset.interleave(
        lambda *args: tf.data.Dataset.from_tensor_slices(args),
        cycle_length=n_worker,
        block_length=1,
        num_parallel_calls=n_worker
    ) # size of dataset = number of windows

    if batch_size is not None:
        dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


class BaselineDataWrapper(Dataset):
    '''
    Wrap create_tf_dataset into a pytorch dataset.
    __getitem__ returns a batch of EEG and a batch of speech features
    __len__ returns the number of windows
    '''

    def __init__(
        self, 
        data_root,
        split,
        win_sec=5,
        hop_sec=1,
        features=['eeg', 'envelope'],
        feature_dims=(64, 1),
        feature_rates=(64, 64),
        number_mismatch=4,
        batch_size=64,
        n_worker=16
    ):
        super().__init__()
        
        self.kwargs = {
            'data_root': data_root,
            'split': split,
            'win_sec': win_sec, 
            'hop_sec': hop_sec,
            'features': features,
            'feature_dims': feature_dims,
            'feature_rates': feature_rates,
            'number_mismatch': number_mismatch,
            'batch_size': batch_size,
            'n_worker': n_worker
        }

        self.reload()

        # Test run to count the number of batches
        self.n_batch_total = 0
        for batch in tqdm(self.tf_loader, desc='Counting batches in TF dataset'):
            self.n_batch_total += 1
            
        self.reload()
        
    def reload(self):
        # Recreate tf dataset
        self.tf_loader = iter(
            create_tf_dataset(
                **self.kwargs
            )
        )

    def __len__(self):
        return self.n_batch_total

    def __getitem__(self, idx):
        # idx is depreciated
        # eeg is always the first returned
        eeg, speech_feature = self.tf_loader.next()

        # (B, M, C, T)
        eeg = torch.tensor(eeg.numpy())\
            .permute(0, 1, 3, 2).contiguous()
        speech_feature = torch.tensor(speech_feature.numpy())\
            .permute(0, 1, 3, 2).contiguous()
        
        return eeg, speech_feature