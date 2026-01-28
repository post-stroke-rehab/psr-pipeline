import numpy as np
import pandas as pd
from scipy.signal import welch
import matplotlib.pyplot as plt

def extract_features(segments, fs=1000, threshold=0.01, wamp_threshold=0.05,
                     window_stride=0.2, start_time=0.0, plot_feature=None):
    '''
    Compute sEMG time- and frequency-domain features for each window and
    return a full feature tensor with timestamps for ML use.

    Parameters
    ----------
    segments : np.ndarray or list[np.ndarray]
        Either:
            - 2D array (n_channels, n_samples): single continuous signal split internally, OR
            - list of 1D sEMG windows (each window pre-segmented).
    fs : int
        Sampling frequency (Hz).
    threshold : float
        Threshold for zero crossings / slope sign changes.
    wamp_threshold : float
        Threshold for Willison Amplitude.
    window_stride : float
        Time between consecutive windows (s).
    start_time : float
        Timestamp of first window.
    plot_feature : str or None
        If provided, plots this feature (e.g. 'RMS') vs. time.

    Returns
    -------
    feature_tensor : np.ndarray
        (n_windows, n_features) array of feature values.
    feature_df : pd.DataFrame
        Feature DataFrame with timestamps.
    timestamps : np.ndarray
        Center time of each window.
    '''

    # --- Helper for per-window feature extraction ---
    def compute_window_features(x):
        x = np.asarray(x).flatten()
        dx = np.diff(x)

        # --- Time domain ---
        RMS  = np.sqrt(np.mean(x ** 2))
        MAV  = np.mean(np.abs(x))
        IEMG = np.sum(np.abs(x))
        WL   = np.sum(np.abs(np.diff(x)))
        VAR  = np.var(x)
        ZC   = np.sum(((x[:-1] * x[1:]) < 0) & (np.abs(x[:-1] - x[1:]) > threshold))
        SSC  = np.sum(((dx[:-1] * dx[1:]) < 0) & (np.abs(dx[:-1] - dx[1:]) > threshold))
        WAMP = np.sum(np.abs(dx) > wamp_threshold)

        # --- Frequency domain ---
        f, Pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
        Pxx = Pxx / np.sum(Pxx + 1e-12)
        MNF = np.sum(f * Pxx)
        cumulative_power = np.cumsum(Pxx)
        MDF = f[np.where(cumulative_power >= 0.5)[0][0]]
        SEN = -np.sum(Pxx * np.log2(Pxx + 1e-12))
        TP  = np.sum(Pxx)

        return {
            'RMS': RMS, 'MAV': MAV, 'IEMG': IEMG, 'WL': WL, 'VAR': VAR,
            'ZC': ZC, 'SSC': SSC, 'WAMP': WAMP,
            'MNF': MNF, 'MDF': MDF, 'SEN': SEN, 'TP': TP
        }

    # --- Handle input shape ---
    if isinstance(segments, np.ndarray) and segments.ndim == 2:
        # If continuous data provided (n_channels, n_samples), assume already windowed externally
        segments = [segments[i, :] for i in range(segments.shape[0])]
    elif isinstance(segments, np.ndarray) and segments.ndim == 1:
        segments = [segments]  # single window

    # --- Extract features per window ---
    feature_list = []
    timestamps = []

    for i, seg in enumerate(segments):
        feats = compute_window_features(seg)
        feature_list.append(feats)
        timestamps.append(start_time + i * window_stride)

    # --- Assemble into DataFrame and Tensor ---
    feature_df = pd.DataFrame(feature_list)
    feature_df['timestamp'] = timestamps
    feature_df = feature_df[['timestamp'] + [c for c in feature_df.columns if c != 'timestamp']]

    feature_tensor = feature_df.drop(columns=['timestamp']).to_numpy()
    timestamps = np.array(timestamps)

    # --- Optional plotting ---
    if plot_feature is not None and plot_feature in feature_df.columns:
        plt.figure(figsize=(8, 4))
        plt.plot(feature_df['timestamp'], feature_df[plot_feature], '-o')
        plt.title(f'{plot_feature} Feature Trajectory')
        plt.xlabel('Time (s)')
        plt.ylabel(plot_feature)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return feature_tensor, feature_df, timestamps

def extract_features_multichannel(windows: np.ndarray, fs: float) -> np.ndarray:
    """
    Extract features for windowed multi-channel EMG.

    Parameters
    ----------
    windows : np.ndarray
        Shape (n_windows, window_samples, n_channels)
    fs : float
        Sampling frequency

    Returns
    -------
    X : np.ndarray
        Shape (n_windows, n_channels * n_features_per_channel)
        Feature order: for each channel in increasing index order,
        [RMS, MAV, IEMG, WL, VAR, ZC, SSC, WAMP, MNF, MDF, SEN, TP]
    """
    if not isinstance(windows, np.ndarray):
        windows = np.array(windows)

    if windows.ndim != 3:
        raise ValueError(f"windows must be 3D (n_windows, window_samples, n_channels). Got {windows.shape}")

    n_windows, window_samples, n_channels = windows.shape

    #ise existing single-signal extractor on each channel-window
    #only need the tensor output; timestamps/df are irrelevant here.
    feature_rows = []

    for i in range(n_windows):
        feats_per_ch = []
        for ch in range(n_channels):
            #one 1D window for one channel
            x_1d = windows[i, :, ch]
            ft, _, _ = extract_features(x_1d, fs=int(fs), window_stride=1.0, start_time=0.0)
            #ft shape is (1, 12) because single window
            feats_per_ch.append(ft[0])
        #concatenate channel feature vectors -> (n_channels*12,)
        feature_rows.append(np.concatenate(feats_per_ch, axis=0))

    return np.vstack(feature_rows)
