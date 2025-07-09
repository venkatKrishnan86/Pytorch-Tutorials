# Reference: https://github.com/genisplaja/diffusion-vocal-sep/blob/master/augmentation_utils.py

import numpy as np
import librosa
import torch
import torchaudio
from torchaudio.transforms import PitchShift, TimeStretch

def _pitch_shift(single_audio, sample_rate, _shift):
    """
        Arguments
        ---------
        single_audio: torch.Tensor
            The audio tensor - Shape: (channels, audio_length)

        sample_rate: int
            The sample rate of the audio

        _shift: int
            The number of steps to shift the audio

        Returns
        -------
        shifted_audio: torch.Tensor
            The shifted audio tensor - Shape: (channels, audio_length)
    """
    pitch_shifter = PitchShift(sample_rate=sample_rate, n_steps=_shift)
    return pitch_shifter(single_audio)

def _time_stretch(single_audio, _stretch):
    """
        Arguments
        ---------
        single_audio: torch.Tensor
            The audio tensor - Shape: (channels, audio_length)

        _stretch: float
            The stretch factor

        Returns
        -------
        strectched_audio: torch.Tensor
            The stretched audio tensor - Shape: (channels, audio_length)
    """
    time_stretcher = TimeStretch(hop_length=256, n_freq=513)
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=None)(single_audio)
    stretched_spectrogram = time_stretcher(spectrogram, _stretch)
    strectched_audio = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256)(stretched_spectrogram)
    
    return strectched_audio

def pitch_augment(mixture, vocal, sample_rate):
    """
        Arguments
        ---------
        mixture: torch.Tensor
            The mixture audio tensor - Shape: (num_examples, channels, audio_length)

        vocal: torch.Tensor
            The vocal audio tensor - Shape: (num_examples, channels, audio_length)

        sample_rate: int
            The sample rate of the audio

        
        Returns
        -------
        mixture: torch.Tensor
            The pitch shifted mixture audio tensor - Shape: (num_examples, channels, audio_length)

        vocal: torch.Tensor
            The pitch shifted vocal audio tensor - Shape: (num_examples, channels, audio_length)
    """
    num_examples = mixture.shape[0]
    shift_values = (torch.rand((num_examples,)) * 8) - 4    # Range: [-4, 4]
    mixture = torch.stack(
        [_pitch_shift(mixture[i], sample_rate, shift_values[i]) for i in range(num_examples)]
    )
    vocal = torch.stack(
        [_pitch_shift(vocal[i], sample_rate, shift_values[i]) for i in range(num_examples)]
    )
    return mixture, vocal

def time_augment(mixture, vocal, accomp):
    """
        Arguments
        ---------
        mixture: torch.Tensor
            The mixture audio tensor - Shape: (num_examples, channels, audio_length)

        vocal: torch.Tensor
            The vocal audio tensor - Shape: (num_examples, channels, audio_length)

        accomp: torch.Tensor
            The accompaniment audio tensor - Shape: (num_examples, channels, audio_length)


        Returns
        -------
        mixture: torch.Tensor
            The stretched mixture audio tensor - Shape: (num_examples, channels, stretched_audio_length)

        vocal: torch.Tensor
            The stretched vocal audio tensor - Shape: (num_examples, channels, stretched_audio_length)

        accomp: torch.Tensor
            The stretched accompaniment audio tensor - Shape: (num_examples, channels, stretched_audio_length)
    """
    num_examples = mixture.shape[0]
    shift_values = (torch.rand((num_examples,)) * 1.25) + 0.5   # Range: [0.5, 1.75]
    mixture = [_time_stretch(mixture[i], shift_values[i]) for i in range(num_examples)]
    vocal = [_time_stretch(vocal[i], shift_values[i]) for i in range(num_examples)]
    accomp = [_time_stretch(accomp[i], shift_values[i]) for i in range(num_examples)]
    return mixture, vocal, accomp

def phase_vocoder(
    single_audio: torch.Tensor, 
    n_fft=1024,
    hop_len=256, 
    rate=0.8
):
    """
        Arguments
        ---------
        single_audio: torch.Tensor
            The audio tensor - Shape: (channels, audio_length)
        hop_len: int
            The hop length for the phase vocoder
        rate: float
            The rate for the phase vocoder (>0)
        
        Returns
        -------
        D_stretched: torch.Tensor
            The stretched complex valued spectrogram tensor - Shape: (new_num_frames, num_bins)
    """
    D = librosa.stft(single_audio.numpy(), n_fft=n_fft, hop_length=hop_len)
    D_stretched = librosa.phase_vocoder(D, hop_length=hop_len, rate=rate)
    new_audio = librosa.istft(D_stretched, hop_length=hop_len)
    return torch.Tensor(new_audio)

def phase_augment(mixture, vocal, accomp):
    """
        Arguments
        ---------
        mixture: torch.Tensor
            The mixture audio tensor - Shape: (num_examples, channels, audio_length)

        vocal: torch.Tensor
            The vocal audio tensor - Shape: (num_examples, channels, audio_length)

        accomp: torch.Tensor
            The accompaniment audio tensor - Shape: (num_examples, channels, audio_length)


        Returns
        -------
        mixture: torch.Tensor
            The phase stretched mixture audio tensor - Shape: (num_examples, channels, stretched_audio_length)

        vocal: torch.Tensor
            The phase stretched vocal audio tensor - Shape: (num_examples, channels, stretched_audio_length)

        accomp: torch.Tensor
            The phase stretched accompaniment audio tensor - Shape: (num_examples, channels, stretched_audio_length)
    """
    num_examples = mixture.shape[0]
    shift_values = (torch.rand((num_examples,)) * 1.25) + 0.5   # Range: [0.5, 1.75]
    mixture = [phase_vocoder(mixture[i], hop_len=256, rate=shift_values[i]) for i in range(num_examples)]
    vocal = [phase_vocoder(vocal[i], hop_len=256, rate=shift_values[i]) for i in range(num_examples)]
    accomp = [phase_vocoder(accomp[i], hop_len=256, rate=shift_values[i]) for i in range(num_examples)]
    return mixture, vocal, accomp

if __name__ == "__main__":
    num_frames = 100
    num_bins = 1025
    hop_len = 256
    rate = 0.8

    # Example Spectrogram Tensor with complex values
    D = torch.randn(num_frames, num_bins, dtype=torch.complex64)

    D_stretched = phase_vocoder(D, hop_len=256, rate=0.8)
    print(D_stretched.shape)