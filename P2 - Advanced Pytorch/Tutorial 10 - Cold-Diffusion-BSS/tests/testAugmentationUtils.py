import unittest
import torch
import numpy as np
import sys
sys.path.append('../')
from augmentation_utils import _pitch_shift, _time_stretch, pitch_augment, time_augment, phase_vocoder

class TestAugmentationUtils(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 16000
        self.audio_length = 32000
        self.channels = 2
        self.single_audio = torch.randn(self.channels, self.audio_length)
        self.mixture = torch.randn(5, self.channels, self.audio_length)
        self.vocal = torch.randn(5, self.channels, self.audio_length)
        self.accomp = torch.randn(5, self.channels, self.audio_length)
        self.stft_tensor = torch.randn(100, 1025, dtype=torch.complex64)

    def test_pitch_shift(self):
        shifted_audio = _pitch_shift(self.single_audio, self.sample_rate, 2)
        self.assertEqual(shifted_audio.shape, self.single_audio.shape)

    def test_time_stretch(self):
        stretch = np.random.rand()+0.5
        stretched_audio = _time_stretch(self.single_audio, stretch)
        self.assertLessEqual(np.abs(stretched_audio.shape[1] - self.single_audio.shape[1]//stretch), 150)

    def test_pitch_augment(self):
        mixture_aug, vocal_aug, accomp_aug = pitch_augment(self.mixture, self.vocal, self.accomp, self.sample_rate)
        self.assertEqual(mixture_aug.shape, self.mixture.shape)
        self.assertEqual(vocal_aug.shape, self.vocal.shape)
        self.assertEqual(accomp_aug.shape, self.accomp.shape)

    def test_time_augment(self):
        mixture_aug, vocal_aug, accomp_aug = time_augment(self.mixture, self.vocal, self.accomp)
        self.assertEqual(mixture_aug[0].shape, vocal_aug[0].shape)
        self.assertEqual(mixture_aug[1].shape, vocal_aug[1].shape)
        self.assertEqual(mixture_aug[0].shape, accomp_aug[0].shape)
        self.assertEqual(mixture_aug[1].shape, accomp_aug[1].shape)

    def test_phase_vocoder(self):
        D_stretched = phase_vocoder(self.stft_tensor, hop_len=256, rate=0.8)
        self.assertEqual(D_stretched.shape, self.stft_tensor.shape)

if __name__ == "__main__":
    unittest.main()