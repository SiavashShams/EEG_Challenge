
'''
Copied and modified from brainpipe
'''

"""Code to calculate speech envelopes."""

import numpy as np
from brian2 import Hz
from brian2hears import Sound, erbspace, Gammatone, Filterbank

from brain_pipe.pipeline.base import PipelineStep


class EnvelopeFromGammatone(Filterbank):
    """Converts the output of a GammatoneFilterbank to an envelope."""

    def __init__(self, source, power_factor):
        """Initialize the envelope transformation.

        Parameters
        ----------
        source : Gammatone
            Gammatone filterbank output to convert to envelope
        power_factor : float
            The power factor for each sample.
        """
        super().__init__(source)
        self.power_factor = power_factor
        self.nchannels = 1

    def buffer_apply(self, input_):  # noqa: D102
        return np.reshape(
            np.sum(np.power(np.abs(input_), self.power_factor), axis=1, keepdims=True),
            (np.shape(input_)[0], self.nchannels),
        )


class GammatoneEnvelope():
    """Calculates a gammatone envelope."""

    def __init__(
        self,
        power_factor=0.6,
        sr=16000,
        min_freq=50,
        max_freq=5000,
        bands=28,
    ):
        """Initialize the gammatone envelope FeatureExtractor.

        Parameters
        ----------
        power_factor : float
            The power factor for each sample
        """
        super(GammatoneEnvelope, self).__init__()
        self.power_factor = power_factor
        self.sr = sr
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.bands = bands

    def __call__(self, wav):
        """Calculate the gammatone envelope.

        Parameters
        ----------
        data_dict: dict
            Dictionary containing the stimulus data and sampling rate.

        Returns
        -------
        dict
            Dictionary containing the gammatone envelope.
        """
        sound = Sound(wav, samplerate=self.sr * Hz)
        # 28 center frequencies from 50 Hz till 5kHz
        center_frequencies = erbspace(
            self.min_freq * Hz, self.max_freq * Hz, self.bands
        )
        filter_bank = Gammatone(sound, center_frequencies)
        envelope_calculation = EnvelopeFromGammatone(filter_bank, self.power_factor)
        envelope = envelope_calculation.process()

        return envelope
