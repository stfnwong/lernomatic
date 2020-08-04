"""
AUDIO_FEATURE
Extract/manipulate audio features

Stefan Wong 2018
"""

import librosa
import scipy
import subprocess
import numpy as np

# debug
#


# Feature extactors
# I'm heavily taking ideas from the dcase_utils repo
class FeatureExtractor(object):
    """
    FeatureExtractor()
    Base class for all feature extractors.

    Args:
        fs  (int)  : Sample frequency (Hz)
        eps (float): Epsilon

    """
    def __init__(self, **kwargs) -> None:
        self.fs                 : int      = kwargs.pop('fs', 44100)
        self.eps                : float    = kwargs.pop('eps', 1e-8)
        self.win_length_samples : int      = kwargs.pop('win_length_samples', 2048)        # window length in samples
        self.hop_length_samples : int      = kwargs.pop('hop_length_samples', None)        # hop length in samples
        self.win_length_seconds : int      = kwargs.pop('win_length_seconds', None)        # window length in seconds
        self.hop_length_seconds : int      = kwargs.pop('hop_length_seconds', None)        # hop length in seconds
        self.dtype              : np.float = kwargs.pop('dtype', np.float)

        if self.win_length_samples is None and self.win_length_seconds is not None:
            self.win_length_samples = int(self.fs * self.win_length_seconds)
        if self.hop_length_samples is None and self.hop_length_seconds is not None:
            self.hop_length_samples = int(self.fs * self.hop_length_seconds)

    def __repr__(self) -> str:
        return 'FeatureExtractor'

    def __str__(self) -> str:
        s = []
        s.append('FeatureExtractor %.4fHz\n' % self.fs)
        return ''.join(s)

    def __getstate__(self) -> dict:
        return {
            'eps' : self.eps,
            'fs'  : self.fs,
            'win_length_samples' : self.win_length_samples,
            'win_length_seconds' : self.win_length_seconds,
            'hop_length_samples' : self.hop_length_samples,
            'hop_length_seconds' : self.hop_length_seconds
        }

    def __setstate(self, state: dict) -> None:
        self.eps                = state['eps']
        self.fs                 = state['fs']
        self.win_length_samples = state['win_length_samples']
        self.win_length_seconds = state['win_length_seconds']
        self.hop_length_samples = state['hop_length_samples']
        self.hop_length_seconds = state['hop_length_seconds']

    def get_window_function(self, win_len: int, win_type: str):     # TODO: return type hint here?
        if win_type == 'hamming_asymmetric':
            return scipy.signal.hamming(win_len, sym=False)
        elif win_type == 'hamming_symmetric':
            return scipy.signal.hamming(win_len, sym=True)
        elif win_type == 'hann_asymmetric':
            return scipy.signal.hann(win_len, sym=False)
        elif win_type == 'hann_symmetric':
            return scipy.signal.hann(win_len, sym=True)
        else:
            raise ValueError('Unknown window function [%s]' % str(win_type))

    def get_spectogram(self,
                       X: np.ndarray,
                       n_fft: int,
                       win_length_samples: int ,
                       hop_length_samples: int ,
                       spect_type: str,
                       center,
                       window) -> np.ndarray:
        if spect_type == 'mag':
            return np.abs(librosa.stft(
                X + self.eps,
                n_fft = n_fft,
                win_length = win_length_samples,
                hop_length = hop_length_samples,
                center = center,
                window = window)
            )
        elif spect_type == 'pow':
            return np.abs(librosa.sftf(
                X + self.eps,
                n_fft = n_fft,      # <- why was this missing before?
                win_length = win_length_samples,
                hop_length = hop_length_samples,
                center = center,
                window = window)
            ) ** 2
        else:
            raise ValueError('Unsupported spect_type [%s]' % str(spect_type))

    def set_hop_len_seconds(self, hop_len:float) -> None:
        self.hop_length_samples = int(self.fs * hop_len)

    def set_hop_len_samples(self, hop_len:int) -> None:
        self.hop_length_samples = hop_len

    def set_win_len_seconds(self, win_len:float) -> None:
        self.win_length_samples = int(self.fs * win_len)

    def set_win_len_samples(self, win_len:int) -> None:
        self.win_length_samples = win_len

    def get_feature_shape(self) -> tuple:
        raise NotImplementedError('Implement this method in derived class')

    def extract(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Implement this method in derived class')


# Extract Mel features with librosa
class MelSpectogramExtractor(FeatureExtractor):
    def __init__(self, **kwargs) -> None:
        # TODO : documentation
        self.n_mels         : int  = kwargs.pop('n_mels', 40)
        self.fmin           : int  = kwargs.pop('fmin', 0)
        self.fmax           : int  = kwargs.pop('fmax', 22000)
        self.n_fft          : int  = kwargs.pop('nfft', 2048)
        self.win_type       : str  = kwargs.pop('window_type', 'hamming_asymmetric')
        self.spect_type     : str  = kwargs.pop('spect_type', 'mag')
        self.logarithmic    : bool = kwargs.pop('logarithmic', True)
        self.norm_mel_bands : bool = kwargs.pop('norm_mel_bands', False)
        self.htk            : bool = kwargs.pop('htk', False)

        # pass on the rest of keyword args
        super(MelSpectogramExtractor, self).__init__(**kwargs)

        self.mel_basis = librosa.filters.mel(
            sr     = self.fs,
            n_fft  = self.n_fft,
            n_mels = self.n_mels,
            fmin   = self.fmin,
            fmax   = self.fmax,
            htk    = self.htk
        )
        if self.norm_mel_bands:
            self.mel_basis /= np.max(self.mel_basis, axis=-1)[:, None]

    def __repr__(self) -> str:
        return 'MelSpectogramExtractor'

    def __str__(self) -> str:
        s = []
        s.append('MelSpectogramExtractor (%.4fHz -> %4.fHz)\n' % (self.fmin, self.fmax))
        s.append('%d mel bands, %d FFT bins\n' % (self.n_mels, self.n_fft))

        return ''.join(s)

    def __getstate__(self) -> dict:
        state = super(MelSpectogramExtractor, self).__getstate__()
        state.update({
            'n_mels'         : self.n_mels,
            'fmin'           : self.fmin,
            'fmax'           : self.fmax,
            'norm_mel_bands' : self.norm_mel_bands,
            'htk'            : self.htk,
            'logarithmic'    : self.logarithmic,
        })

        return state

    def __setstate__(self, state: dict) -> None:
        super(MelSpectogramExtractor, self).__setstate__(state)

    def get_spect_param(self) -> dict:
        # Intended for setting HDF5 attributes rather than serialization
        return {
            'hop_length'  : self.hop_length_samples,
            'win_lengths' : self.win_length_samples,
            'win_type'    : self.win_type,
            'spect_type'  : self.spect_type,
            'sample_rate' : self.fs,
            'num_fft'     : self.n_fft,
            'num_mel'     : self.n_mels,
            'log'         : self.logarithmic,
            'htk'         : self.htk,
            'fmin'        : self.fmin,
            'fmax'        : self.fmax,
            'norm_mel'    : self.norm_mel_bands,
            'fmin'        : self.fmin,
            'fmin'        : self.fmin,
        }

    def get_feature_shape(self, slen:int) -> tuple:
        return (int(self.n_mels), int(1 + np.ceil(slen / self.hop_length_samples)))

    def get_req_hop_size(self, slen:int, req_t:int) -> int:
        return int(np.ceil(slen / req_t))

    def get_req_sample_len(self, req_t:int) -> int:
        return int(self.hop_length_samples  * (req_t - 1))

    def extract(self, X: np.ndarray) -> np.ndarray:
        """
        extract()

        Extract the features from the data item X
        """
        window = self.get_window_function(self.win_length_samples, self.win_type)
        spectogram = self.get_spectogram(
            X,
            n_fft = self.n_fft,
            win_length_samples = self.win_length_samples,
            hop_length_samples = self.hop_length_samples,
            spect_type = self.spect_type,
            center = True,
            window = window
        )

        mel_spect = np.dot(self.mel_basis, spectogram)
        if self.logarithmic:
            mel_spect = np.log(mel_spect + self.eps)

        return mel_spect


class MFCCFeatureExtractor(FeatureExtractor):
    """
    MFCCFeatureExtractor()
    Extracts Mel-Frequency Cepstral Coeffcient features.

    Arguments:
        n_mels (int): Number of mel bands
        fmin   (int): Minimum frequency (Hz)
        fmin   (int): Maximum frequency (Hz)
        n_fft  (int): Number of FFT bands

    """
    def __init__(self, **kwargs) -> None:
        self.n_mels         = kwargs.pop('n_mels', 40)
        self.fmin           = kwargs.pop('fmin', 0)
        self.fmax           = kwargs.pop('fmax', 22000)
        self.n_fft          = kwargs.pop('nfft', 2048)
        self.win_type       = kwargs.pop('window_type', 'hamming_asymmetric')
        self.spect_type     = kwargs.pop('spect_type', 'mag')
        self.logarithmic    = kwargs.pop('logarithmic', True)
        self.norm_mel_bands = kwargs.pop('norm_mel_bands', False)
        self.htk            = kwargs.pop('htk', False)
        self.n_mfcc         = kwargs.pop('n_mfcc', 20)
        self.omit_zeroth    = kwargs.pop('omit_zeroth', False)

        super(MFCCFeatureExtractor, self).__init__(**kwargs)

        self.mel_basis = librosa.filters.mel(
            sr     = self.fs,
            n_fft  = self.n_fft,
            n_mels = self.n_mels,
            fmin   = self.fmin,
            fmax   = self.fmax,
            htk    = self.htk
        )
        if self.norm_mel_bands:
            self.mel_basis /= np.max(self.mel_basis, axis=-1)[:, None]

    def __repr__(self) -> str:
        return 'MFCCFeatureExtractor'

    def __str__(self) -> str:
        s = []
        s.append('MFCCFeatureExtractor (%.4fHz -> %4.fHz)\n' % (self.fmin, self.fmax))
        s.append('%d mel bands, %d FFT bins\n' % (self.n_mels, self.n_fft))

        return ''.join(s)


# ==== Other ===== #
class Phonify(object):
    """
    PHONIFY
    This is basically a copy of the phonify class from the DCASE repo.
    Converts dB SPL to phon units by applying the Terhardt outer ear
    transfer function. The full scale signal dB SPL equivalent needs to
    be given on initialization (default : 96dB)
    """
    def __init__(self, frqs, db_max=96, bias=1e-8, clip=True):
        self.freqs = frqs
        self.to_db = lambda x : self.lin2db(x, bias)
        self.corr  = self.terhardt_db(frqs) - self.terhardt_db(1000) + db_max
        self.clip  = clip

    @staticmethod
    def lin2db(x, bias=1e-8):
        x = x + bias
        np.log10(x, out=x)
        x = x * 20.0

        return x

    @staticmethod
    def terhardt_db(f):
        fk = f / 1000
        tdb = -3.64 * fk ** -0.8 + 6.5 * np.exp(-0.6 * (fk-3.3)**2) - 1.e-33*(fk)**4
        return tdb

    def __call__(self, frames, out=None):
        frames = np.asarray(frames)
        spec = self.to_db(frames)

        if out is None:
            out = np.empty_like(spec)
        out[:] = spec
        out += self.corr        # applies the outer-ear transfer function
        if self.clip:
            np.maximum(out, 0.0, out=out)       # clip values below the listening threshold

        return out      # dB SPL


class Resampler(object):
    def __init__(self, **kwargs) -> None:
        """
        RESAMPLER
        Change sampling rate. Note that this does not attempt
        to resample from memory, rather it reads the source file
        from disk and calls out to a subprocess to perform the
        resampling operation.
        """
        self.sample_rate : int        = kwargs.pop('sample_rate', 22050)
        self.use_ffmpeg  : bool       = kwargs.pop('use_ffmpeg', True)
        self.downmix     : bool       = kwargs.pop('downmix', False)
        self.dtype       : np.float32 = kwargs.pop('dtype', np.float32)
        # internal
        self.verbose     : bool       = kwargs.pop('verbose', False)

        if self.use_ffmpeg is True:
            self.cmd = 'ffmpeg'
        self.probe = self.cmd[:2] + 'probe'

    def __repr__(self) -> str:
        return 'Resampler-%d' % int(self.sample_rate)

    def __str__(self) -> str:
        return self.__repr__()

    def _get_num_channels(self, fname: str, debug=False) -> int:
        call = [self.probe, "-v", "quiet", "-show_streams", fname]
        if debug is True:
            return call
        info = subprocess.check_output(call)
        for l in info.split():
            if l.startswith(bytes('channels=', 'ascii')):
                return int(l[len('channels='):])
        return 0

    def _construct_call(self, fname: str) -> None:
        call = [self.cmd, "-v", "quiet", "-y", \
                "-i", fname, "-f", "f32le", \
                "-ar", str(self.sample_rate), "pipe:1"]
        return call

    def resample(self, sample_filename: str) -> np.ndarray:
        call = self._construct_call(sample_filename)
        if self.downmix:
            call[8:8] = ["ac", "1"]
        else:
            num_channels = self._get_num_channels(sample_filename)
        samples = subprocess.check_output(call)
        samples = np.frombuffer(samples, dtype=self.dtype)
        if self.downmix is False:
            samples = samples.reshape((num_channels, -1), order='F')

        return samples
