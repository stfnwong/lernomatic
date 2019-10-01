"""
READ_WAV
Utilities for reading *.wav files

Stefan Wong 2018
"""

import wave
import numpy as np


def read_wave(filename:str,
              downmix:bool=True,
              to_float:bool = True,
              return_meta:bool=False) -> np.ndarray:
    f = wave.open(filename)
    sample_rate = f.getframerate()
    nchannels   = f.getnchannels()
    nsamples    = f.getnframes()
    samples     = f.readframes(nsamples)
    sampwidth   = f.getsampwidth()

    f.close()

    # Convert to ndarray
    samples = np.frombuffer(samples, dtype=np.int16)
    if to_float is True:
        samples = (samples / 2.0 ** 15).astype(np.float32)

    if downmix is False:
        # just split the interleaved samples into one row per channel
        samples = samples.reshape((nchannels, -1), order='F')
    elif nchannels == 2:
        samples = (samples[::2] + samples[1: :2]) / 2
    elif nchannels > 2:
        raise ValueError('Unsupported wave file %s. Only mono or stereo streams supported' % str(filename))

    if return_meta:
        meta = {
            'nchannels': nchannels,
            'framerate': sample_rate,
            'nframes'  : nsamples,
            'sampwidth': sampwidth
        }
        return (samples, meta)

    return samples


def write_wav(filename :str, data:np.ndarray, **kwargs) -> None:
    meta         = kwargs.pop('meta', None)
    num_samples  = kwargs.pop('num_samples', None)
    sample_rate  = kwargs.pop('sample_rate', 44100)
    num_channels = kwargs.pop('num_channels', 1)
    sample_width = kwargs.pop('sample_width', 2)        # unit here is bytes

    if num_samples is None and meta is None:
        num_samples = len(data)
    elif num_samples is None and meta is not None:
        num_samples = meta['nframes']

    fp = wave.open(filename, 'w')
    fp.setnframes(num_samples)
    if meta is not None:
        fp.setnchannels(meta['nchannels'])
        fp.setframerate(meta['framerate'])
        fp.setsampwidth(meta['sampwidth'])
    else:
        fp.setnchannels(num_channels)
        fp.setframerate(sample_rate)
        fp.setsampwidth(sample_width)

    fp.writeframes(data)
    fp.close()
