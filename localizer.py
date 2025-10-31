import numpy as np
import mne
from mne.minimum_norm import read_inverse_operator, apply_inverse_raw

# Load pre-computed inverse operator
inv_operator = read_inverse_operator('inverse_operator-inv.fif', verbose=False)

info = mne.io.read_info("info.fif", verbose=False)


def localize(data: np.ndarray, stop: int | None = None) -> np.ndarray:
    # data shape should be (n_channels, n_times)
    raw = mne.io.RawArray(data, info, verbose=False)
    # Set average reference (required)
    # try:
    #     raw.set_eeg_reference('average', projection=True, verbose=False)
    # except:
    #     pass

    # Apply inverse to raw data
    stc = apply_inverse_raw(raw, inv_operator, 
                            lambda2=1.0/9.0, 
                            method='dSPM',
                            buffer_size=5000,  # process in chunks
                            start=0, 
                            stop=stop,
                            verbose=False)  # first 60 seconds

    return stc.data