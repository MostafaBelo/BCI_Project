from hdf5storage import loadmat
import localizer
import region_filter
import numpy as np
from tqdm import tqdm

import os

for file in tqdm(os.listdir("/mnt/C/BCI Dataset/Task1_Matlab/Matlab files/")):
    subject_name = file.strip("results").strip("_SR.mat")
    if subject_name == "ZAB":
        continue
    print(subject_name)

    mat = loadmat(
        f"/mnt/C/BCI Dataset/Task1_Matlab/Matlab files/{file}")
    res = np.empty((1, 400), dtype=object)

    for i in tqdm(range(400)):
        eeg_signal = mat["sentenceData"]["rawData"][0, i]
        try:
            localized_eeg = localizer.localize(eeg_signal)
            filter_local_eeg = region_filter.filter_signal(localized_eeg)
            res[0, i] = filter_local_eeg
        except Exception as e:
            print(f"Error processing sentence {i}: {e}")
            print(eeg_signal.shape)
            res[0, i] = None

    os.makedirs(f"shards/{subject_name}", exist_ok=True)
    for i in range(4):
        np.savez(f"shards/{subject_name}/shard_{i}.npz",
                 *(res[:, i*100:(i+1)*100]))
