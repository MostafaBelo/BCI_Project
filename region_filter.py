import numpy as np
import json
import os

data_dir = "/mnt/D/University/Fall 2025/BCI/Project/shards/words/"
# data_dir = "shards"

with open("language_roi_config.json", "r") as f:
    data = json.load(f)
    roi_indexes = data["roi_vertices"]

for shard_idx in range(1):
    shard_path = os.path.join(
        data_dir, f"old/shard_{shard_idx}.npy")
    shard_path_out = os.path.join(
        data_dir, f"shard_{shard_idx}.npy")
    shard = np.load(shard_path, allow_pickle=True)

    for i in range(len(shard)):
        if shard[i] is None:
            continue
        for j in range(len(shard[i]["localized_eeg"])):
            if shard[i]["localized_eeg"][j] is None:
                continue
        shard[i]["localized_eeg"][j] = shard[i]["localized_eeg"][j][roi_indexes, :]
    np.save(shard_path_out, shard)
