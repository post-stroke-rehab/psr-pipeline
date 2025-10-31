def segment_signal(signal, fs, window_size=0.2, overlap=0.5):
    '''
    Segment a recorded sEMG signal into overlapping windows.

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed sEMG signal (filtered + denoised).
    fs : int or float
        Sampling frequency (Hz).
    window_size : float
        Window length in seconds, default 0.2 (200 ms).
    overlap : float
        Fraction of overlap between consecutive windows, default 0.5.

    Returns
    -------
    segments : np.ndarray
        Array of shape (num_windows, window_samples).
    timestamps : np.ndarray
        Array of [t_start, t_center, t_end] for each window (seconds).
    '''
    pass

