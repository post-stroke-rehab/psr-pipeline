import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfilt

def bandpass_filter(signal, fs: float, low: float = 20.0, high: float = 450.0, order: int = 4, axis: int = -1, zero_phase: bool = True) -> np.ndarray: 
    """
    Butterworth band-pass filter on raw sEMG

    Parameters
    ----------
    signal : np.ndarray
        1D or ND array. Filtering is along `axis`
    fs : float
        Sampling frequency (Hz), must be > 0
    low : float
        Low cutoff (Hz), default 20
    high : float
        High cutoff (Hz), default 450 (must be < fs/2)
    order : int
        Filter order (>=1). Higher -> sharper rolloff
    axis : int
        Axis to filter along (default last axis) but basically the "direction" of time in inputted signals
    zero_phase : bool
        If True, use forward-backward (zero-phase) filtering
        If False, use causal filtering for shorter windows w/o enough padding but introduces phase lag

        
    Returns
    -------
    np.ndarray
        Filtered signal outputted in same shape as input


    ValueError
        On invalid parameters or when zero-phase padding is not possible
        for very short signals. In that case, consider `zero_phase=False`,
        a smaller `order`, or trimming/extending the signal.
    """
    x = np.asarray(signal, float)

    # if fs <= 0:
    #     raise ValueError("fs must be positive.")
    # if order <= 0:
    #     raise ValueError("order must be a positive integer")
    if low <= 0:
        raise ValueError("low must be > 0 Hz.")

    nyq = 0.5 * fs
    if not low < high < nyq:
        raise ValueError(
            f"Cutoffs must satisfy 0 < low < high < Nyquist ({nyq:.3f} Hz). "
            f"Got low={low}, high={high}, fs={fs}.")
    if x.size == 0:
        return x  # nth to filter

    #convertin cutoff frequencies from Hz to 0-1 scale
    wn = [low / nyq, high / nyq]

    #designing a butterworth band-pass in SOS form
    sos = butter(order, wn, btype="bandpass", output="sos")
    #print(sos.shape) #(order_sections, 6)
  
    if zero_phase:
        # callers can choose to (a) lower order, (b) pad/trim, or (c) set zero_phase=False.
        return sosfiltfilt(sos, x, axis, 'odd', None) 
    else:
        return sosfilt(sos, x, axis)

