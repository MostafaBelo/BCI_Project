import numpy as np
import json
import os

from tqdm import tqdm

data_dir = "/mnt/D/University/Fall 2025/BCI/Project/shards/words/"
# data_dir = "shards"

with open("language_roi_config.json", "r") as f:
    data = json.load(f)
    roi_indexes = data["roi_vertices"]


def filter_signal(signal):
    return signal[roi_indexes, :]


if __name__ == "__main__":
    for shard_idx in tqdm(range(8)):
        shard_path = os.path.join(
            data_dir, f"old/shard_{shard_idx}.npy")
        shard_path_out = os.path.join(
            data_dir, f"shard_{shard_idx}.npy")
        shard = np.load(shard_path, allow_pickle=True)

        for i in range(len(shard)):
            # for i in range(49):
            if shard[i] is None:
                continue
            for j in range(len(shard[i]["localized_eeg"])):
                if shard[i]["localized_eeg"][j] is None:
                    continue
                shard[i]["localized_eeg"][j] = filter_signal(
                    shard[i]["localized_eeg"][j])
        shard = [item for item in shard if (item is not None)]
        np.save(shard_path_out, shard)
