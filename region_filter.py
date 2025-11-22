import numpy as np
import json
import os

# data_dir = "/home/mostafaelfaggal/Documents/BCI/"
data_dir = "shards"

with open("language_roi_config.json", "r") as f:
    data = json.load(f)
    roi_indexes = data["roi_vertices"]

for shard_idx in range(8):
    shard_path = os.path.join(
        data_dir, f"localized_eeg_sentences_ZAB_shard{shard_idx}.npy")
    shard_path_out = os.path.join(
        data_dir, f"localized_eeg_sentences_ZAB_shard{shard_idx}_out.npy")
    shard = np.load(shard_path, allow_pickle=True)

    for i in range(len(shard)):
        if shard[i] is None:
            continue
        shard[i] = shard[i][roi_indexes, :]
    np.save(shard_path_out, shard)
