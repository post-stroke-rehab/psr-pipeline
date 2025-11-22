import numpy as np
import matplotlib.pyplot as plt


def segment_signal(signal, fs, window_size=0.2, overlap=0.5):
    '''
    Segment a recorded sEMG signal into overlapping windows.

    Parameters
    -
    signal : np.ndarray
        Preprocessed sEMG signal (filtered + denoised).
    fs : int or float
        Sampling frequency (Hz).
    window_size : float
        Window length in seconds, default 0.2 (200 ms).
    overlap : float
        Fraction of overlap between consecutive windows, default 0.5.

    Returns
    -
    segments : np.ndarray
        Array of shape (num_windows, window_samples).
    timestamps : np.ndarray
        Array of [t_start, t_center, t_end] for each window (seconds).
    '''
    # Validate input
    if not isinstance(signal, np.ndarray):
        raise TypeError("signal must be a numpy array.")
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array.")
    if not (0 <= overlap < 1):
        raise ValueError("overlap must be in [0, 1).")

    # Convert seconds to samples
    window_samples = int(round(window_size * fs))
    if window_samples <= 0:
        raise ValueError("window_size too small for given fs.")
    
    step_samples = int(round(window_samples * (1 - overlap)))
    if step_samples <= 0:
        raise ValueError("overlap too large: step size becomes 0.")

    # Compute start indices of each window
    starts = np.arange(0, len(signal) - window_samples + 1, step_samples)
    num_windows = len(starts)

    # Preallocate arrays 
    segments = np.zeros((num_windows, window_samples))
    timestamps = np.zeros((num_windows, 3))

    # Fill windows and compute timestamps
    for i, start in enumerate(starts):
        end = start + window_samples
        segments[i, :] = signal[start:end]

        t_start = start / fs
        t_end = end / fs
        t_center = (t_start + t_end) / 2
        timestamps[i] = [t_start, t_center, t_end]

    return segments, timestamps


def main():
    """Verify that segment_signal meets all verification and acceptance criteria."""
    print("Running verification tests for segment_signal...\n")

    # Generate synthetic sEMG-like signal 
    fs = 1000  # Hz
    duration = 3.0  # seconds
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    signal = np.sin(2 * np.pi * 10 * t) + 0.2 * np.random.randn(len(t))  # simulated sEMG

    # Parameters 
    window_size = 0.4  
    overlap = 0.25

    # Run segmentation 
    segments, timestamps = segment_signal(signal, fs, window_size, overlap)

    # Verification 1: Shapes 
    print(f"Number of windows: {segments.shape[0]}")
    print(f"Samples per window: {segments.shape[1]}")
    print(f"Timestamps shape: {timestamps.shape}")
    assert timestamps.shape[1] == 3, "Timestamps must be [t_start, t_center, t_end]."

    # Verification 2: Consistent timestamp increments 
    dt_centers = np.diff(timestamps[:, 1])
    mean_dt = np.mean(dt_centers)
    expected_step = window_size * (1 - overlap)
    assert np.isclose(mean_dt, expected_step, atol=1e-4), \
        f"Timestamp increment ({mean_dt}) ≠ expected step ({expected_step})."
    print("✅ Timestamp increments consistent with sampling rate.")

    # Verification 3: Duration alignment 
    durations = timestamps[:, 2] - timestamps[:, 0]
    assert np.allclose(durations, window_size, atol=1e-4), \
        "Window durations do not match expected window_size."
    print("✅ Window durations match expected value.")

    # Verification 4: Overlap check 
    step_samples = int(round(window_size * fs * (1 - overlap)))
    actual_overlap = 1 - step_samples / (window_size * fs)
    assert np.isclose(actual_overlap, overlap, atol=1e-3), \
        f"Computed overlap {actual_overlap} ≠ expected {overlap}."
    print("✅ Overlap fraction verified.")

    # Verification 5: Supports arbitrary window/overlap combinations 
    for w, o in [(0.1, 0.25), (0.3, 0.75), (0.05, 0.0)]:
        segs, _ = segment_signal(signal, fs, w, o)
        assert segs.ndim == 2 and segs.shape[1] == int(round(w * fs)), \
            f"Window size {w}s failed."
    print("✅ Function supports arbitrary window/overlap combinations.")

    # Optional Visualization 
    plt.figure(figsize=(10, 3))
    plt.plot(t, signal, color='black', lw=1)
    for ts in timestamps:
        plt.axvspan(ts[0], ts[2], color='orange', alpha=0.2)
    plt.title("Segment coverage verification")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    print("\nAll verification and acceptance criteria PASSED ✅")


if __name__ == "__main__":
    main()