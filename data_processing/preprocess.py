import numpy as np
from .bandpass_filter import bandpass_filter
from .wavelet_denoise import batch_wavelet_denoise
from .windowing import segment_multichannel
from .feature_extraction import extract_features_multichannel
from .preprocess_config import PreprocessConfig


def preprocess_emg(emg: np.ndarray, fs: float, config: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    """
    Full preprocessing pipeline:
    bandpass -> wavelet denoise -> windowing -> feature extraction

    Input:  emg shape (n_samples, n_channels)
    Output: features shape (n_windows, n_channels * 12) by default
            or windows shape (n_windows, window_samples, n_channels) if config.return_windows=True
    """
    emg = np.asarray(emg, float)

    #bandpass filter along time axis
    emg = bandpass_filter(
        emg,
        fs=fs,
        low=config.low,
        high=config.high,
        order=config.order,
        axis=0,
        zero_phase=config.zero_phase
    )

    #wavelet denoising per channel
    emg = batch_wavelet_denoise(
        emg,
        wavelet=config.wavelet,
        level=config.level,
        threshold_mode=config.threshold_mode,
        threshold_method=config.threshold_method,
        noise_est=config.noise_est
    )

    #windowing
    windows, _ = segment_multichannel(
        emg,
        fs=fs,
        window_size=config.window_size,
        overlap=config.overlap,
        padding=config.padding
    )

    if config.return_windows:
        return windows

    # feature extraction
    X = extract_features_multichannel(windows, fs=fs)

    #checking output dimensions
    expected_dim = windows.shape[2] * 12
    assert X.shape[1] == expected_dim, f"Feature dim mismatch: got {X.shape[1]}, expected {expected_dim}"

    return X
