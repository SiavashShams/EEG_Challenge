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


@tf.function
def batch_equalizer_fn(*args):
    """Batch equalizer.
    Prepares the inputs for a model to be trained in
    match-mismatch task. It makes sure that match_env
    and mismatch_env are equally presented as a first
    envelope in match-mismatch task.

    Parameters
    ----------
    args : Sequence[tf.Tensor]
        List of tensors representing feature data

    Returns
    -------
    Tuple[Tuple[tf.Tensor], tf.Tensor]
        Tuple of the EEG/speech features serving as the input to the model and
        the labels for the match/mismatch task

    Notes
    -----
    This function will also double the batch size. E.g. if the batch size of
    the elements in each of the args was 32, the output features will have
    a batch size of 64.
    """
    eeg = args[0]
    num_stimuli = len(args) - 1
    # repeat eeg num_stimuli times
    new_eeg = tf.concat([eeg] * num_stimuli, axis=0)
    all_features = [new_eeg]

    # create args
    args_to_zip = [args[i::num_stimuli] for i in range(1,num_stimuli+1)]
    for stimuli_features in zip(*args_to_zip):

        for i in range(num_stimuli):
            stimulus_rolled = tf.roll(stimuli_features, shift=i, axis=0)
            # reshape stimulus_rolled to merge the first two dimensions
            stimulus_rolled = tf.reshape(stimulus_rolled, [tf.shape(stimulus_rolled)[0] * tf.shape(stimulus_rolled)[1], stimuli_features[0].shape[-2], stimuli_features[0].shape[-1]])

            all_features.append(stimulus_rolled)
    labels = tf.concat(
        [
            tf.tile(tf.constant([[1 if ii == i else 0 for ii in range(num_stimuli)]]), [tf.shape(eeg)[0], 1]) for i in range(num_stimuli)
        ], axis=0
    )

    return tuple(all_features), labels


def shuffle_fn(args, number_mismatch):
    # repeat the last argument number_ mismatch times
    args = list(args)
    for _  in range(number_mismatch):
        args.append(tf.random.shuffle(args[-1]))
    return tuple(args)


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
    batch_equalizer='all'
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
        num_parallel_calls=tf.data.AUTOTUNE
    ) # (n_win, win_len, 64 or 1)

    if number_mismatch is not None:
        # map second argument (speech) to shifted version
        dataset = dataset.map(lambda *args : shuffle_fn(args, number_mismatch),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.interleave(
        lambda *args: tf.data.Dataset.from_tensor_slices(args),
        cycle_length=8,
        block_length=1,
        num_parallel_calls=tf.data.AUTOTUNE
    ) # size of dataset = number of windows

    if batch_size is not None:
        dataset = dataset.batch(batch_size, drop_remainder=True)
        # (B, T1, C1) and (B, T2, C2)

    if batch_equalizer == 'all':
        # Create the labels and make sure classes are balanced
        dataset = dataset.map(batch_equalizer_fn,
                              num_parallel_calls=tf.data.AUTOTUNE)

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
        batch_equalizer='all'
    ):
        super().__init__()
        
        assert batch_equalizer in ['all', 'random']
        self.batch_equalizer = batch_equalizer

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
            'batch_equalizer': batch_equalizer
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

        # Same as the baseline. Expand the batch_size by (number_mismatch+1).
        if self.batch_equalizer == 'all':

            features, label = self.tf_loader.next()
            eeg = features[0]
            speech_feature = tf.stack(features[1:], axis=1)

            # (B*M, T, C) -> (B*M, C, T)
            eeg = torch.tensor(eeg.numpy())\
                .permute(0, 2, 1).contiguous()
            # (B*M, M, C, T)
            speech_feature = torch.tensor(speech_feature.numpy())\
                .permute(0, 1, 3, 2).contiguous()
                
            label = torch.argmax(torch.tensor(label.numpy()), dim=1)
        
            return eeg, speech_feature, label

        elif self.batch_equalizer == 'random':

            features = self.tf_loader.next()
            eeg = features[0]
            speech_feature = tf.stack(features[1:], axis=1)

            # (B, T, C) -> (B, C, T)
            eeg = torch.tensor(eeg.numpy())\
                .permute(0, 2, 1).contiguous()
            # (B, M, C, T)
            speech_feature = torch.tensor(speech_feature.numpy())\
                .permute(0, 1, 3, 2).contiguous()
                
            # The matched speech is at the first of all 5 windows.
            # Randomly shuffle all 5 windows, so that the matched speech
            # is not always the first one.
            B, M = speech_feature.shape[0], speech_feature.shape[1]
            perm_indices = [torch.randperm(M) for _ in range(B)]
            speech_feature = torch.stack(
                [speech_feature[i, perm_indices[i]] for i in range(B)],
                dim=0
            )
            label = torch.tensor(
                [torch.argsort(perm_indices[i])[0] for i in range(B)]
            )
        
            return eeg, speech_feature, label