"""
TEST_FEATURE_EXTRACTOR
Test the (librosa) feature extraction objects

Stefan Wong 2019

"""

import sys
import argparse
import unittest
import numpy as np
import matplotlib.pyplot as plt
from lernomatic.util import wav_util
# units under test
from lernomatic.feature import audio_feature
from lernomatic.feature import audio_loop

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()

def get_figure():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    return fig, ax


# make a new dataset for the test
def create_dataset(data_root:str,
                   out_file:str,
                   max_elem:int=256) -> None:
    pass

class TestMelSpectogramExtractor(unittest.TestCase):
    def setUp(self):
        self.verbose     = GLOBAL_OPTS['verbose']
        self.input_file  = GLOBAL_OPTS['test_data_root'] + '/' + GLOBAL_OPTS['test_data_file']
        self.sample_rate = 44100
        self.test_win_length = 2048    # unit is samples
        self.test_hop_length = 512    # unit is samples
        self.test_hop_length_seconds = 1
        self.test_num_mel_bands = 80

    def test_extract_feature(self):
        print('======== TestMelSpectogramExtractor.test_extract_feature ')

        print('Reading sample from file %s' % self.input_file)
        sample = wav_util.read_wave(self.input_file, self.sample_rate)
        print('len(sample), sample rate : %.6f : %d' % (len(sample), self.sample_rate))

        # use all defaults for this test
        fextract = audio_feature.MelSpectogramExtractor(
            win_length_samples = self.test_win_length,
            hop_length_samples = self.test_hop_length,
            n_mels = self.test_num_mel_bands
        )
        print(fextract)

        # Ensure that the parameters in the super class were set correctly
        self.assertEqual(True, hasattr(fextract, 'win_length_samples'))
        self.assertEqual(True, hasattr(fextract, 'hop_length_samples'))
        self.assertEqual(True, hasattr(fextract, 'win_length_seconds'))
        self.assertEqual(True, hasattr(fextract, 'hop_length_seconds'))
        self.assertEqual(self.test_win_length, fextract.win_length_samples)
        self.assertEqual(self.test_hop_length, fextract.hop_length_samples)

        print('Generating mel spectogram feature for file [%s]' % self.input_file)
        mel_spect = fextract.extract(sample)
        print('Feature shape : %s' % str(mel_spect.shape))
        fig, ax = get_figure()
        ax.imshow(mel_spect)
        ax.set_title('STFT output for file %s' % self.input_file)
        ax.set_xlabel('Samples')
        ax.set_ylabel('Bank')
        fig.savefig('figures/test_mel_spect_extractor_mel_spect.png', bbox_inches='tight')

        if GLOBAL_OPTS['draw_plot']:
            plt.show()

        print('======== TestMelSpectogramExtractor.test_extract_feature <END>')

    def test_feature_shape(self):
        print('======== TestMelSpectogramExtractor.test_feature_shape ')

        print('Reading sample from file %s' % self.input_file)
        sample, sample_meta = wav_util.read_wave(self.input_file, return_meta=True)
        print('len(sample), sample rate : %.6f : %d' % (len(sample), sample_meta['framerate']))

        # use all defaults for this test
        fextract = audio_feature.MelSpectogramExtractor(
            win_length_samples = self.test_win_length,
            hop_length_samples = self.test_hop_length,
            n_mels = self.test_num_mel_bands
        )
        print(fextract)
        print('predicted feature shape : %s' % str(fextract.get_feature_shape(len(sample))))
        mel_spect = fextract.extract(sample)
        print('mel_spect shape : %s' % str(mel_spect.shape))

        print('======== TestMelSpectogramExtractor.test_feature_shape <END>')

    def test_hop_window_seconds(self):
        print('======== TestMelSpectogramExtractor.test_hop_window_seconds ')

        print('Reading sample from file %s' % self.input_file)
        sample, sample_meta = wav_util.read_wave(self.input_file, return_meta=True)
        print('len(sample), sample rate : %.6f : %d' % (len(sample), sample_meta['framerate']))

        # use all defaults for this test
        test_window_len_seconds = 0.04
        test_hop_len_seconds = 0.01
        fextract = audio_feature.MelSpectogramExtractor(
            win_length_samples = 2048,
            fs = sample_meta['framerate'],
            n_fft = 2048,
            win_length_seconds = test_window_len_seconds,
            hop_length_seconds = test_hop_len_seconds,
            n_mels = self.test_num_mel_bands
        )
        print(fextract)
        pred_feature_shape = fextract.get_feature_shape(len(sample))
        print('predicted feature shape : %s' % str(pred_feature_shape))
        mel_spect = fextract.extract(sample)
        print('mel_spect shape : %s' % str(mel_spect.shape))
        self.assertEqual(len(pred_feature_shape), len(mel_spect.shape))
        for n in range(len(pred_feature_shape)):
            self.assertEqual(pred_feature_shape[n], mel_spect.shape[n])

        print('======== TestMelSpectogramExtractor.test_hop_window_seconds <END>')

    def test_stft_shape_prediction(self):
        print('======== TestMelSpectogramExtractor.test_stft_shape_prediction ')

        print('Reading sample from file %s' % self.input_file)
        sample, sample_meta = wav_util.read_wave(self.input_file, return_meta=True)
        print('len(sample), sample rate : %d : %d' % (len(sample), sample_meta['framerate']))

        # use all defaults for this test
        req_t = 1001
        test_window_len_seconds = 0.04
        test_hop_len_seconds = 0.01
        fextract = audio_feature.MelSpectogramExtractor(
            fs = sample_meta['framerate'],
            n_fft = 2048,
            win_length_seconds = test_window_len_seconds,
            hop_length_seconds = test_hop_len_seconds,
            n_mels = self.test_num_mel_bands
        )
        pred_feature_shape = fextract.get_feature_shape(len(sample))
        print('predicted feature shape : %s' % str(pred_feature_shape))
        mel_spect = fextract.extract(sample)
        print('mel_spect shape : %s' % str(mel_spect.shape))

        self.assertEqual(mel_spect.shape, pred_feature_shape)
        self.assertEqual(req_t, mel_spect.shape[1])

        # what should hop_size be to get an output shape of 512?
        req_t = 512
        pred_hop_size = fextract.get_req_hop_size(len(sample), req_t)
        print('Predicted hop size : %d' % pred_hop_size)
        fextract.set_hop_len_samples(pred_hop_size)
        pred_feature_shape = fextract.get_feature_shape(len(sample))
        print('predicted feature shape : %s' % str(pred_feature_shape))
        mel_spect = fextract.extract(sample)
        print('mel_spect shape : %s' % str(mel_spect.shape))

        self.assertEqual(req_t, mel_spect.shape[1])

        print('======== TestMelSpectogramExtractor.test_stft_shape_prediction <END>')

    def test_set_spectogram_t(self):
        print('======== TestMelSpectogramExtractor.test_set_spectogram_t ')
        # test that we can adjust the hop size such that the output has a given
        # shape along the t axis

        print('Reading sample from file %s' % self.input_file)
        sample, sample_meta = wav_util.read_wave(self.input_file, return_meta=True)
        print('len(sample), sample rate : %d : %d' % (len(sample), sample_meta['framerate']))

        # use all defaults for this test
        req_t = 2000
        test_window_len_seconds = 0.04
        test_hop_len_seconds = 0.01
        fextract = audio_feature.MelSpectogramExtractor(
            fs = sample_meta['framerate'],
            n_fft = 2048,
            win_length_seconds = test_window_len_seconds,
            hop_length_seconds = test_hop_len_seconds,
            n_mels = self.test_num_mel_bands
        )

        # how long does the sample need to be to get an output where t=2000?
        pred_sample_len = fextract.get_req_sample_len(req_t)
        print('Predicted sample length should be %d for output spectogram to have %d time samples' %\
              (pred_sample_len, req_t)
        )

        # get a looper and loop up the sample
        looper = audio_loop.Looper(pred_sample_len)
        looped_sample = looper.loop(sample)
        print('looped_sample shape : %s' % str(looped_sample.shape))

        mel_spect = fextract.extract(looped_sample)
        pred_feature_shape = fextract.get_feature_shape(len(looped_sample))
        print('predicted feature shape : %s' % str(pred_feature_shape))
        print('mel_spect shape : %s' % str(mel_spect.shape))
        self.assertEqual(len(pred_feature_shape), len(mel_spect.shape))
        for n in range(len(pred_feature_shape)):
            self.assertEqual(pred_feature_shape[n], mel_spect.shape[n])


        print('======== TestMelSpectogramExtractor.test_set_spectogram_t <END>')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',
                        default=True,
                        type=bool,
                        help='Sets verbose mode'
                        )
    parser.add_argument('--draw-plot',
                        default=False,
                        type=bool,
                        help='Draw plots'
                        )
    # Location of test data, etc
    parser.add_argument('--test-data-prefix',
                        type=str,
                        default='/mnt/ml-data/datasets/freesound-kaggle/',
                        help='Prefix for datasets used in unit test '
                        )
    parser.add_argument('--test-data-root',
                        type=str,
                        default='./test/test_data',
                        #default='/mnt/ml-data/datasets/kaggle/FSD/FSDKaggle2018.audio_test',
                        help='Path to root of test data'
                        )
    parser.add_argument('--test-data-file',
                        type=str,
                        default='34363.wav',
                        help='Filename of test data file'
                        )
    # spectogram options
    parser.add_argument('--num-filters',
                        type=int,
                        default=80,
                        help='Number of filters to use in filterbank'
                        )
    parser.add_argument('--frame-len',
                        type=int,
                        default=2048,
                        help='Spectogram frame length'
                        )
    parser.add_argument('--frame-rate',
                        type=int,
                        default=100,
                        help='Spectogram frame rate in Hz'
                        )
    parser.add_argument('--mag-scale',
                        type=str,
                        choices = ('log', 'power', 'phon', 'none'),
                        default='log',
                        help='Type of magnitude scale to use in Filterbank. Options are mel, log, lin (default: mel)'
                        )
    parser.add_argument('--freq-scale',
                        type=str,
                        choices=('mel', 'log', 'lin'),
                        default='mel',
                        help='Type of frequency scale to use in Spectogram. Options are log, power, phon (default: log)'

                        )
    parser.add_argument('--filter-shape',
                        type=str,
                        choices = ('tri', 'hann'),
                        default = 'tri',
                        help='Window shape for filterbank. Options are tri, hann (default: tri)'
                        )
    parser.add_argument('--dtype',
                        type=str,
                        choices = ('float32', 'float64'),
                        default='float64',
                        help='Data type to use for filterbank. Options are float32, float64 (default: float64)'
                        )
    parser.add_argument('--min-freq',
                        type=float,
                        default=27.5,
                        help='Minimum frequency for filterbank'
                        )
    parser.add_argument('--max-freq',
                        type=float,
                        default=16000.0,
                        help='Maximum frequency for filterbank'
                        )
    parser.add_argument('--sample-rate',
                        type=int,
                        default=44100,
                        help='Sample rate of input samples (in Hz)'
                        )
    parser.add_argument('unittest_args', nargs='*')

    args = parser.parse_args()
    arg_vals = vars(args)
    for k, v in arg_vals.items():
        GLOBAL_OPTS[k] = v

    if GLOBAL_OPTS['verbose']:
        print(' ======== GLOBAL OPTIONS ======== ')
        for k, v in GLOBAL_OPTS.items():
            print('\t[%s] : %s' % (str(k), str(v)))

    sys.argv[1:] = args.unittest_args
    unittest.main()
