def bandpass_filter(signal, fs, low=20, high=450, order=4):
    '''
    Apply a Butterworth band-pass filter to the raw sEMG signal.

    Parameters
    ----------
    signal : np.ndarray
        Raw sEMG signal.
    fs : int or float
        Sampling frequency (Hz).
    low : float
        Low cutoff frequency (Hz), default 20.
    high : float
        High cutoff frequency (Hz), default 450.
    order : int
        Filter order controlling roll-off sharpness, default 4.

    Returns
    -------
    np.ndarray
        Filtered signal with noise and motion artifacts reduced.
    '''
    pass
