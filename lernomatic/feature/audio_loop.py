"""
AUDIO_LOOP
Looping utilities for samples

Stefan Wong 2019
"""

import numpy as np


class SampleLooper(object):
    def __init__(self, required_len_samples:int) -> None:
        self.required_len_samples  :int   = int(required_len_samples)

    def __repr__(self) -> str:
        return 'SampleLooper'

    def loop(self, sample:np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Looper(SampleLooper):
    def __init__(self, required_len_samples:int, **kwargs) -> None:
        super(Looper, self).__init__(required_len_samples)

    def __repr__(self) -> str:
        return 'Looper'

    def loop(self, sample:np.ndarray, return_num_loops:bool=False) -> np.ndarray:
        # basic operation is just to concat and clip
        if len(sample.shape) == 1:
            sample_axis = 0
        else:       # multi-channel sample
            sample_axis = 1

        if len(sample) > self.required_len_samples:
            return sample[0 : self.required_len_samples]

        if (2 * len(sample)) > self.required_len_samples:
            out_sample = np.concatenate((sample, sample), axis=None)
        else:
            N = 1
            out_sample = np.concatenate((sample, sample), axis=None)
            while (N * len(sample)) < self.required_len_samples:
                N += 1
                out_sample = np.concatenate((out_sample, out_sample), axis=None)

        if return_num_loops:
            return (out_sample[0 : self.required_len_samples], N)

        return out_sample[0 : self.required_len_samples]


class XFadeLooper(SampleLooper):
    def __init__(self, required_len_samples:int, **kwargs) -> None:
        self.fade_len :int = kwargs.pop('fade_len', 44100)
        super(XFadeLooper, self).__init__(required_len_samples)

    def __repr__(self) -> str:
        return 'XFadeLooper'

    def rising_triangle(self, N:int) -> np.ndarray:
        T = np.ones(N)
        for n in range(1, len(T)):
            T[n] = n / len(T)

        return T

    def falling_triangle(self, N:int) -> np.ndarray:
        T = np.ones(N)
        for n in range(1, len(T)):
            T[n] = 1.0 - (n / len(T))

        return T

    def xfade_join(self, A:np.ndarray, B:np.ndarray, sample_axis:int) -> np.ndarray:
        a_fade = np.zeros_like(A)
        b_fade = np.zeros_like(B)

        a_fade[0 : len(A) - self.fade_len] = A[0 : len(A) - self.fade_len]
        a_fade[len(A) - self.fade_len :]   = A[len(A) - self.fade_len] * self.falling_triangle(self.fade_len)
        b_fade[self.fade_len : len(B)] = B[self.fade_len : len(B)]
        b_fade[0 : self.fade_len] = B[0 : self.fade_len] * self.rising_triangle(self.fade_len)
        # mix the two middle parts
        mix = np.add(a_fade[len(A) - self.fade_len :], b_fade[0 : self.fade_len])
        X = np.concatenate((a_fade[0 : len(A) - self.fade_len], mix, b_fade[self.fade_len :]), axis=sample_axis)

        #a_fade[len(A) - self.fade_len :] = A[len(A) - self.fade_len :] * self.falling_triangle(self.fade_len)
        #b_fade[0 : self.fade_len] = B[0 : self.fade_len] * self.rising_triangle(self.fade_len)
        #X = np.concatenate((a_fade, b_fade[self.fade_len:]), axis=sample_axis)
        #X[len(A) : len(A) + self.fade_len] += b_fade[0 : self.fade_len]

        return X

    def loop(self, sample:np.ndarray, return_num_loops:bool=False) -> np.ndarray:
        if len(sample.shape) == 1:
            sample_axis = 0
        else:       # multi-channel sample
            sample_axis = 1

        if len(sample) > self.required_len_samples:
            return sample[0 : self.required_len_samples]

        if (2 * len(sample) - self.fade_len) > self.required_len_samples:
            out_sample = self.xfade_join(sample, sample, None)
        else:
            N = 1
            out_sample = self.xfade_join(sample, sample, None)
            while (N * len(sample)) < self.required_len_samples:
                N += 1
                out_sample = self.xfade_join(out_sample, out_sample, None)

        if return_num_loops:
            return (out_sample[0 : self.required_len_samples], N)

        return out_sample[0 : self.required_len_samples]
