import librosa
import numpy as np


def audio_to_spectrogram(audio, n_fft=2048, hop_length=256):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    return magnitude, phase


def normalize_spectrogram(magnitude):
    log_mag = np.log1p(magnitude)
    return (log_mag - log_mag.mean()) / log_mag.std()


def spectrogram_to_audio(magnitude, phase, n_fft=4096, hop_length=512):
    mag = np.expm1(magnitude)
    stft = mag * (np.cos(phase) + 1j * np.sin(phase))
    return librosa.istft(stft, hop_length=hop_length)
