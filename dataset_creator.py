from hdf5storage import loadmat
import localizer
import region_filter
import numpy as np
from tqdm import tqdm

import os

# data_dir = "/mnt/C/BCI Dataset/Task1_Matlab/Matlab files"
data_dir = "/home/g4/Documents/BCI_Project/Dataset/Task1_Matlab/Matlab files"

labels = "shards/labels.txt"

with open(labels, "r") as f:
    labels = [line.split(":", 2)
              for line in f.read().strip().split("\n")]
    labels = [(int(idx), int(sentiment)+1, sentence)
              for idx, sentiment, sentence in labels]

for file in tqdm(os.listdir(data_dir)):
    subject_name = file.strip("results").strip("_SR.mat")
    # if subject_name == "ZAB":
    #     continue
    print(subject_name)

    mat = loadmat(os.path.join(data_dir, file))
    res = np.empty((1, 400), dtype=object)

    for i in tqdm(range(400)):
        eeg_signal = mat["sentenceData"]["rawData"][0, i]
        _, sent, txt = labels[i]
        try:
            localized_eeg = localizer.localize(eeg_signal)
            filter_local_eeg = region_filter.filter_signal(localized_eeg)

            # data = eeg_signal
            data = filter_local_eeg

            res[0, i] = (data, sent, txt)
        except Exception as e:
            print(f"Error processing sentence {i}: {e}")
            print(eeg_signal.shape)
            res[0, i] = (None, None, None)

    # os.makedirs(f"shards/{subject_name}", exist_ok=True)
    for i in range(4):
        np.savez(f"shards/shard_{i}_{subject_name}.npz",
                 *(res[:, i*100:(i+1)*100]))
