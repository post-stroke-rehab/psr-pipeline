from dataclasses import dataclass

@dataclass(frozen=True)
class PreprocessConfig:
    #bandpass
    low: float = 20.0
    high: float = 450.0
    order: int = 4
    zero_phase: bool = True

    #wavelet
    wavelet: str = "sym4"
    level: int = 4
    threshold_mode: str = "soft"
    threshold_method: str = "universal"
    noise_est: str = "median"

    #windowing
    window_size: float = 0.2
    overlap: float = 0.5
    padding: bool = False

    #output features
    return_windows: bool = False
