from hdf5storage import loadmat
import localizer
import numpy as np
from tqdm import tqdm
import os

mat = loadmat("/mnt/C/BCI Dataset/Task1_Matlab/Matlab files/resultsZAB_SR.mat")

out_dir = "shards/words"

os.makedirs(out_dir, exist_ok=True)

sentences_count = mat["sentenceData"].shape[1]

shard_size = 50
for shard_start in range(0, sentences_count, shard_size):
    shard_id = shard_start//shard_size

    all_data = [None]*sentences_count
    for sentence_id in tqdm(range(shard_start, min(shard_start+shard_size, sentences_count))):
        words_count = mat["sentenceData"]["word"][0, sentence_id].shape[1]
        fixations_count = max([0 if f.shape[0] == 0 else f.max().item(
        ) for f in mat["sentenceData"]["word"][0, sentence_id]["fixPositions"][0]])
        all_data[sentence_id] = {
            "eeg": [None]*fixations_count,
            "localized_eeg": [None]*fixations_count,
            "labels": [None]*fixations_count
        }

        for word_id in range(words_count):
            word_data = mat["sentenceData"]["word"][0, sentence_id][0, word_id]
            if len(word_data["nFixations"]) == 0:
                continue
            for i in range(word_data["nFixations"][0, 0]):
                # print(word_data["content"].item())
                # print(word_data["rawEEG"][0,0].shape) # 0, fix_position
                # print(word_data["nFixations"][0,0])
                # print(word_data["fixPositions"][0])

                fix_id = word_data["fixPositions"][0, i]

                all_data[sentence_id]["eeg"][fix_id -
                                             1] = word_data["rawEEG"][0, i]
                try:
                    all_data[sentence_id]["localized_eeg"][fix_id -
                                                           1] = localizer.localize(word_data["rawEEG"][0, i])
                except:
                    all_data[sentence_id]["localized_eeg"][fix_id-1] = None
                all_data[sentence_id]["labels"][fix_id -
                                                1] = word_data["content"].item()
    np.save(os.path.join(out_dir, f"shard_{shard_id}.npy"), all_data)
