"""
TEST_FEATURE_EXTRACTOR
Test the (librosa) feature extraction objects

Stefan Wong 2019

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from lernomatic.util import wav_util
# units under test
from lernomatic.feature import audio_feature
from lernomatic.feature import audio_loop


#def get_figure():
#    fig = plt.figure()
#    ax = fig.add_subplot(1,1,1)
#    return fig, ax
#
#
## make a new dataset for the test
#def create_dataset(data_root:str,
#                   out_file:str,
#                   max_elem:int=256) -> None:
#    pass
#
#
#class TestMelSpectogramExtractor:
#    verbose     = True
#    test_data_prefix='/mnt/ml-data/datasets/freesound-kaggle/'
#    test_data_root ='./test/test_data'
#    test_data_file='34363.wav'
#    input_file  = test_data_root + '/' + test_data_file
#    sample_rate = 44100
#    test_win_length = 2048    # unit is samples
#    test_hop_length = 512    # unit is samples
#    test_hop_length_seconds = 1
#    test_num_mel_bands = 80
#
#    @pytest.mark.skip(reason='Need a way to test this without relying on my data path')
#    def test_extract_feature(self) -> None:
#        print('Reading sample from file %s' % self.input_file)
#        sample = wav_util.read_wave(self.input_file, self.sample_rate)
#        print('len(sample), sample rate : %.6f : %d' % (len(sample), self.sample_rate))
#
#        # use all defaults for this test
#        fextract = audio_feature.MelSpectogramExtractor(
#            win_length_samples = self.test_win_length,
#            hop_length_samples = self.test_hop_length,
#            n_mels = self.test_num_mel_bands
#        )
#        print(fextract)
#
#        # Ensure that the parameters in the super class were set correctly
#        assert hasattr(fextract, 'win_length_samples') is True
#        assert hasattr(fextract, 'hop_length_samples') is True
#        assert hasattr(fextract, 'win_length_seconds') is True
#        assert hasattr(fextract, 'hop_length_seconds') is True
#        assert self.test_win_length == fextract.win_length_samples
#        assert self.test_hop_length == fextract.hop_length_samples
#
#        print('Generating mel spectogram feature for file [%s]' % self.input_file)
#        mel_spect = fextract.extract(sample)
#        print('Feature shape : %s' % str(mel_spect.shape))
#        fig, ax = get_figure()
#        ax.imshow(mel_spect)
#        ax.set_title('STFT output for file %s' % self.input_file)
#        ax.set_xlabel('Samples')
#        ax.set_ylabel('Bank')
#        fig.savefig('figures/test_mel_spect_extractor_mel_spect.png', bbox_inches='tight')
#
#
#    @pytest.mark.skip(reason='Need a way to test this without relying on my data path')
#    def test_feature_shape(self) -> None:
#        print('Reading sample from file %s' % self.input_file)
#        sample, sample_meta = wav_util.read_wave(self.input_file, return_meta=True)
#        print('len(sample), sample rate : %.6f : %d' % (len(sample), sample_meta['framerate']))
#
#        # use all defaults for this test
#        fextract = audio_feature.MelSpectogramExtractor(
#            win_length_samples = self.test_win_length,
#            hop_length_samples = self.test_hop_length,
#            n_mels = self.test_num_mel_bands
#        )
#        print(fextract)
#        print('predicted feature shape : %s' % str(fextract.get_feature_shape(len(sample))))
#        mel_spect = fextract.extract(sample)
#        print('mel_spect shape : %s' % str(mel_spect.shape))
#
#
#    @pytest.mark.skip(reason='Need a way to test this without relying on my data path')
#    def test_hop_window_seconds(self) -> None:
#        print('Reading sample from file %s' % self.input_file)
#        sample, sample_meta = wav_util.read_wave(self.input_file, return_meta=True)
#        print('len(sample), sample rate : %.6f : %d' % (len(sample), sample_meta['framerate']))
#
#        # use all defaults for this test
#        test_window_len_seconds = 0.04
#        test_hop_len_seconds = 0.01
#        fextract = audio_feature.MelSpectogramExtractor(
#            win_length_samples = 2048,
#            fs = sample_meta['framerate'],
#            n_fft = 2048,
#            win_length_seconds = test_window_len_seconds,
#            hop_length_seconds = test_hop_len_seconds,
#            n_mels = self.test_num_mel_bands
#        )
#        print(fextract)
#        pred_feature_shape = fextract.get_feature_shape(len(sample))
#        print('predicted feature shape : %s' % str(pred_feature_shape))
#        mel_spect = fextract.extract(sample)
#        print('mel_spect shape : %s' % str(mel_spect.shape))
#        assert len(pred_feature_shape) == len(mel_spect.shape)
#        for n in range(len(pred_feature_shape)):
#            assert pred_feature_shape[n] == mel_spect.shape[n]
#
#    @pytest.mark.skip(reason='Need a way to test this without relying on my data path')
#    def test_stft_shape_prediction(self) -> None:
#        print('Reading sample from file %s' % self.input_file)
#        sample, sample_meta = wav_util.read_wave(self.input_file, return_meta=True)
#        print('len(sample), sample rate : %d : %d' % (len(sample), sample_meta['framerate']))
#
#        # use all defaults for this test
#        req_t = 1001
#        test_window_len_seconds = 0.04
#        test_hop_len_seconds = 0.01
#        fextract = audio_feature.MelSpectogramExtractor(
#            fs = sample_meta['framerate'],
#            n_fft = 2048,
#            win_length_seconds = test_window_len_seconds,
#            hop_length_seconds = test_hop_len_seconds,
#            n_mels = self.test_num_mel_bands
#        )
#        pred_feature_shape = fextract.get_feature_shape(len(sample))
#        print('predicted feature shape : %s' % str(pred_feature_shape))
#        mel_spect = fextract.extract(sample)
#        print('mel_spect shape : %s' % str(mel_spect.shape))
#
#        assert mel_spect.shape == pred_feature_shape
#        assert req_t == mel_spect.shape[1]
#
#        # what should hop_size be to get an output shape of 512?
#        req_t = 512
#        pred_hop_size = fextract.get_req_hop_size(len(sample), req_t)
#        print('Predicted hop size : %d' % pred_hop_size)
#        fextract.set_hop_len_samples(pred_hop_size)
#        pred_feature_shape = fextract.get_feature_shape(len(sample))
#        print('predicted feature shape : %s' % str(pred_feature_shape))
#        mel_spect = fextract.extract(sample)
#        print('mel_spect shape : %s' % str(mel_spect.shape))
#
#        assert req_t == mel_spect.shape[1]
#
#
#    @pytest.mark.skip(reason='Need a way to test this without relying on my data path')
#    def test_set_spectogram_t(self) -> None:
#        # test that we can adjust the hop size such that the output has a given
#        # shape along the t axis
#        print('Reading sample from file %s' % self.input_file)
#        sample, sample_meta = wav_util.read_wave(self.input_file, return_meta=True)
#        print('len(sample), sample rate : %d : %d' % (len(sample), sample_meta['framerate']))
#
#        # use all defaults for this test
#        req_t = 2000
#        test_window_len_seconds = 0.04
#        test_hop_len_seconds = 0.01
#        fextract = audio_feature.MelSpectogramExtractor(
#            fs = sample_meta['framerate'],
#            n_fft = 2048,
#            win_length_seconds = test_window_len_seconds,
#            hop_length_seconds = test_hop_len_seconds,
#            n_mels = self.test_num_mel_bands
#        )
#
#        # how long does the sample need to be to get an output where t=2000?
#        pred_sample_len = fextract.get_req_sample_len(req_t)
#        print('Predicted sample length should be %d for output spectogram to have %d time samples' %\
#              (pred_sample_len, req_t)
#        )
#
#        # get a looper and loop up the sample
#        looper = audio_loop.Looper(pred_sample_len)
#        looped_sample = looper.loop(sample)
#        print('looped_sample shape : %s' % str(looped_sample.shape))
#
#        mel_spect = fextract.extract(looped_sample)
#        pred_feature_shape = fextract.get_feature_shape(len(looped_sample))
#        print('predicted feature shape : %s' % str(pred_feature_shape))
#        print('mel_spect shape : %s' % str(mel_spect.shape))
#        assert len(pred_feature_shape) == len(mel_spect.shape)
#        for n in range(len(pred_feature_shape)):
#            assert pred_feature_shape[n] == mel_spect.shape[n]
