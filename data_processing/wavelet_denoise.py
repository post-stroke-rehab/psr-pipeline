import numpy as np
import pywt
from typing import Tuple, Optional
import warnings

def wavelet_denoise(signal, wavelet='sym4', level=4, threshold_mode='soft', 
                   threshold_method='universal', noise_est='median'):
    '''
    Perform wavelet-based denoising on filtered sEMG signals.
    
    This implementation uses Discrete Wavelet Transform (DWT) with adaptive 
    thresholding to remove transient noise while preserving activation bursts
    critical for finger movement classification.
    
    Parameters
    ----------
    signal : np.ndarray
        Band-pass filtered sEMG signal (1D array).
    wavelet : str, default='sym4'
        Mother wavelet type. Options: 'sym4', 'db4', 'coif3', etc.
    level : int, default=4
        Decomposition level (higher = more smoothing, but risks over-smoothing).
    threshold_mode : str, default='soft'
        'soft' for gradual suppression, 'hard' for abrupt cutoff.
    threshold_method : str, default='universal'
        Method for threshold calculation: 'universal', 'minimax', or 'sure'.
    noise_est : str, default='median'
        Noise estimation method: 'median' (MAD) or 'std'.
    
    Returns
    -------
    np.ndarray
        Denoised signal with transient noise suppressed and activation preserved.
    
    Notes
    -----
    - Symlet wavelets (sym4) provide good balance between smoothness and localization
    - Level 4 decomposition typically captures sEMG frequency components well
    - Soft thresholding reduces artifacts at activation boundaries
    
    References
    ----------
    [1] Phinyomark et al. (2012). Feature reduction and selection for EMG signal 
        classification. Expert Systems with Applications.
    '''
    
    # Input validation
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)
    
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}")
    
    if len(signal) == 0:
        raise ValueError("Cannot denoise empty signal")
    
    # Check if signal length is sufficient for decomposition
    min_length = 2 ** level
    if len(signal) < min_length:
        warnings.warn(f"Signal length {len(signal)} < minimum {min_length} for level {level}. "
                     f"Reducing level to {int(np.log2(len(signal)))}")
        level = int(np.log2(len(signal)))
    
    # Perform DWT decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Estimate noise level from finest detail coefficients (cD1)
    # Using Median Absolute Deviation (MAD) - robust to outliers
    if noise_est == 'median':
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # MAD estimator
    else:  # 'std'
        sigma = np.std(coeffs[-1])
    
    # Calculate threshold based on selected method
    N = len(signal)
    
    if threshold_method == 'universal':
        # Universal threshold: σ * sqrt(2 * log(N))
        # Guarantees asymptotic minimax properties
        threshold = sigma * np.sqrt(2 * np.log(N))
        
    elif threshold_method == 'minimax':
        # Minimax threshold - more conservative, preserves more signal
        if N > 32:
            threshold = sigma * (0.3936 + 0.1829 * np.log2(N))
        else:
            threshold = 0  # No thresholding for very short signals
            
    elif threshold_method == 'sure':
        # Stein's Unbiased Risk Estimate - adaptive threshold
        threshold = _sure_threshold(coeffs[-1], sigma)
    
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")
    
    # Apply thresholding to detail coefficients (preserve approximation coefficients)
    # We threshold all detail levels but NOT the approximation (coeffs[0])
    coeffs_thresh = [coeffs[0]]  # Keep approximation coefficient unchanged
    
    for i in range(1, len(coeffs)):
        # Apply chosen thresholding mode
        if threshold_mode == 'soft':
            # Soft thresholding: shrinks coefficients gradually
            # Better for smooth signals, reduces artifacts
            coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
            
        elif threshold_mode == 'hard':
            # Hard thresholding: zeros coefficients below threshold
            # Preserves large coefficients exactly
            coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='hard'))
            
        elif threshold_mode == 'garotte':
            # Garrote thresholding: compromise between soft and hard
            coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='garotte'))
        
        else:
            raise ValueError(f"Unknown threshold mode: {threshold_mode}")
    
    # Reconstruct signal from thresholded coefficients
    denoised = pywt.waverec(coeffs_thresh, wavelet)
    
    # Handle potential length mismatch due to wavelet decomposition/reconstruction
    if len(denoised) > len(signal):
        denoised = denoised[:len(signal)]
    elif len(denoised) < len(signal):
        denoised = np.pad(denoised, (0, len(signal) - len(denoised)), mode='edge')
    
    return denoised


def _sure_threshold(detail_coeffs, sigma):
    '''
    Calculate SURE (Stein's Unbiased Risk Estimate) threshold.
    
    This adaptive method minimizes MSE estimate. More complex but can
    outperform universal threshold for certain signal types.
    
    Parameters
    ----------
    detail_coeffs : np.ndarray
        Detail coefficients from wavelet decomposition.
    sigma : float
        Estimated noise standard deviation.
    
    Returns
    -------
    float
        Optimal threshold value.
    '''
    N = len(detail_coeffs)
    abs_coeffs = np.abs(detail_coeffs)
    abs_coeffs_sorted = np.sort(abs_coeffs)
    
    # Calculate SURE risk for different thresholds
    risks = np.zeros(N)
    for k in range(N):
        threshold = abs_coeffs_sorted[k]
        risks[k] = (N - 2 * (k + 1) + 
                   np.sum(np.minimum(abs_coeffs, threshold)**2) + 
                   (k + 1) * threshold**2) / N
    
    # Find threshold that minimizes risk
    min_risk_idx = np.argmin(risks)
    optimal_threshold = abs_coeffs_sorted[min_risk_idx]
    
    return optimal_threshold * sigma


def compare_denoising_methods(signal, fs=2000, wavelets=None, levels=None):
    '''
    Compare different wavelet denoising configurations for sEMG signals.
    
    This utility function helps evaluate which wavelet and decomposition level
    work best for your specific dataset.
    
    Parameters
    ----------
    signal : np.ndarray
        Input sEMG signal (should be band-pass filtered already).
    fs : int, default=2000
        Sampling frequency in Hz.
    wavelets : list of str, optional
        Wavelets to test. Default: ['sym4', 'db4', 'coif3', 'bior3.3']
    levels : list of int, optional
        Decomposition levels to test. Default: [3, 4, 5]
    
    Returns
    -------
    dict
        Performance metrics for each configuration including SNR improvement,
        correlation with original, and RMS error.
    
    Example
    -------
    >>> results = compare_denoising_methods(emg_signal, fs=2000)
    >>> best_config = max(results.items(), key=lambda x: x[1]['snr_improvement'])
    >>> print(f"Best: {best_config[0]} with SNR gain: {best_config[1]['snr_improvement']:.2f} dB")
    '''
    
    if wavelets is None:
        wavelets = ['sym4', 'db4', 'coif3', 'bior3.3']
    
    if levels is None:
        levels = [3, 4, 5]
    
    results = {}
    
    for wavelet in wavelets:
        for level in levels:
            try:
                # Denoise signal
                denoised = wavelet_denoise(signal, wavelet=wavelet, level=level)
                
                # Calculate metrics
                noise = signal - denoised
                
                # SNR improvement (dB)
                signal_power = np.sum(denoised ** 2)
                noise_power = np.sum(noise ** 2)
                snr_improvement = 10 * np.log10(signal_power / (noise_power + 1e-10))
                
                # Correlation with original (should be high)
                correlation = np.corrcoef(signal, denoised)[0, 1]
                
                # RMS error
                rms_error = np.sqrt(np.mean(noise ** 2))
                
                # Waveform integrity: check if activation peaks are preserved
                # This is crucial for sEMG - we need to maintain burst morphology
                peak_preservation = _calculate_peak_preservation(signal, denoised)
                
                key = f"{wavelet}_L{level}"
                results[key] = {
                    'wavelet': wavelet,
                    'level': level,
                    'snr_improvement': snr_improvement,
                    'correlation': correlation,
                    'rms_error': rms_error,
                    'peak_preservation': peak_preservation,
                    'score': snr_improvement * correlation * peak_preservation  # Combined metric
                }
                
            except Exception as e:
                print(f"Failed for {wavelet} level {level}: {e}")
                continue
    
    return results


def _calculate_peak_preservation(original, denoised, percentile=95):
    '''
    Calculate how well activation peaks are preserved after denoising.
    
    For sEMG, preserving the morphology of muscle activation bursts is critical
    for accurate motor intent classification.
    
    Parameters
    ----------
    original : np.ndarray
        Original signal.
    denoised : np.ndarray
        Denoised signal.
    percentile : float, default=95
        Percentile for identifying significant peaks.
    
    Returns
    -------
    float
        Peak preservation score [0, 1], where 1 = perfect preservation.
    '''
    from scipy.signal import find_peaks
    
    # Find peaks in both signals
    threshold_orig = np.percentile(np.abs(original), percentile)
    threshold_denoised = np.percentile(np.abs(denoised), percentile)
    
    peaks_orig, props_orig = find_peaks(np.abs(original), height=threshold_orig)
    peaks_denoised, props_denoised = find_peaks(np.abs(denoised), height=threshold_denoised)
    
    if len(peaks_orig) == 0 or len(peaks_denoised) == 0:
        return 0.0
    
    # Calculate how many original peaks are preserved (within tolerance)
    tolerance = 20  # samples (10ms at 2000Hz)
    matches = 0
    
    for peak_orig in peaks_orig:
        if np.any(np.abs(peaks_denoised - peak_orig) <= tolerance):
            matches += 1
    
    preservation_score = matches / len(peaks_orig)
    
    return preservation_score


def batch_wavelet_denoise(signals, wavelet='sym4', level=4, **kwargs):
    '''
    Apply wavelet denoising to multiple channels simultaneously.
    
    Useful for processing all 12 EMG channels from NinaPro DB2 dataset.
    
    Parameters
    ----------
    signals : np.ndarray
        2D array of shape (n_samples, n_channels).
    wavelet : str, default='sym4'
        Mother wavelet type.
    level : int, default=4
        Decomposition level.
    **kwargs : dict
        Additional arguments passed to wavelet_denoise().
    
    Returns
    -------
    np.ndarray
        Denoised signals with same shape as input.
    
    Example
    -------
    >>> # For NinaPro data with 12 channels
    >>> emg_denoised = batch_wavelet_denoise(emg, wavelet='sym4', level=4)
    '''
    
    if signals.ndim == 1:
        return wavelet_denoise(signals, wavelet, level, **kwargs)
    
    n_samples, n_channels = signals.shape
    denoised = np.zeros_like(signals)
    
    for ch in range(n_channels):
        denoised[:, ch] = wavelet_denoise(signals[:, ch], wavelet, level, **kwargs)
    
    return denoised


# Example usage for your project
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.io as sio
    
    # Load sample data (adjust path as needed)
    # data = sio.loadmat('toy_semg/S1/S1_E1_A1.mat')
    # emg = data['emg']
    
    # For demonstration, create synthetic sEMG-like signal
    fs = 2000  # NinaPro sampling rate
    t = np.arange(0, 2, 1/fs)  # 2 seconds
    
    # Simulate muscle activation burst + noise
    signal_clean = np.zeros_like(t)
    signal_clean[2000:4000] = 50 * np.sin(2 * np.pi * 60 * t[2000:4000])  # 60Hz activation
    noise = 10 * np.random.randn(len(t))  # Gaussian noise
    signal_noisy = signal_clean + noise
    
    # Apply wavelet denoising
    signal_denoised = wavelet_denoise(signal_noisy, wavelet='sym4', level=4)
    
    # Visualize results
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    axes[0].plot(t, signal_clean, 'g-', alpha=0.7, label='Clean')
    axes[0].set_title('Original Clean Signal')
    axes[0].set_ylabel('Amplitude (mV)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t, signal_noisy, 'r-', alpha=0.5, label='Noisy')
    axes[1].set_title('Noisy Signal (Input)')
    axes[1].set_ylabel('Amplitude (mV)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(t, signal_denoised, 'b-', linewidth=1.5, label='Denoised')
    axes[2].set_title('Wavelet Denoised Signal (Output)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude (mV)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('wavelet_denoising_demo.png', dpi=150)
    plt.show()
    
    # Compare different configurations
    print("\n=== Comparing Wavelet Configurations ===")
    results = compare_denoising_methods(signal_noisy, fs=fs)
    
    # Sort by combined score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
    
    print("\nTop 5 Configurations:")
    for i, (config, metrics) in enumerate(sorted_results[:5], 1):
        print(f"\n{i}. {config}")
        print(f"   SNR Improvement: {metrics['snr_improvement']:.2f} dB")
        print(f"   Correlation: {metrics['correlation']:.4f}")
        print(f"   Peak Preservation: {metrics['peak_preservation']:.4f}")
        print(f"   Combined Score: {metrics['score']:.4f}")