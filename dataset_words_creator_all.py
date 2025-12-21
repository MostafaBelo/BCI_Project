from hdf5storage import loadmat
import localizer
import numpy as np
from tqdm import tqdm
import os

data_dir = "/mnt/C/BCI Dataset/Task1_Matlab/Matlab files"
out_dir = "shards/words"
os.makedirs(out_dir, exist_ok=True)

shard_size = 100

labels = "shards/labels.txt"

with open(labels, "r") as f:
    labels = [line.split(":", 2)
              for line in f.read().strip().split("\n")]
    labels = [(int(idx), int(sentiment)+1, sentence)
              for idx, sentiment, sentence in labels]

for file in tqdm(os.listdir(data_dir)):
    subject_name = file.strip("results").strip("_SR.mat")
    print(subject_name)

    mat = loadmat(os.path.join(data_dir, file))

    # shard_id = shard_start//shard_size
    res = np.empty((1, 400), dtype=object)

    for sentence_id in tqdm(range(400)):
        words_count = mat["sentenceData"]["word"][0, sentence_id].shape[1]
        try:
            fixations_count = max([0 if f.shape[0] == 0 else f.max().item(
            ) for f in mat["sentenceData"]["word"][0, sentence_id]["fixPositions"][0]])
        except Exception as e:
            res[0, sentence_id] = {
                "eeg": None,
                "labels": None,
                "sentiment": None,
                "sentence": None
            }
            continue

        res[0, sentence_id] = {
            "eeg": np.empty((fixations_count), dtype=object),
            # "localized_eeg": [None]*fixations_count,
            "labels": [None]*fixations_count,
            "sentiment": labels[sentence_id][1],
            "sentence": labels[sentence_id][2]
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

                res[0, sentence_id]["eeg"][fix_id -
                                           1] = word_data["rawEEG"][0, i]
                # try:
                #     all_data[sentence_id-shard_start]["localized_eeg"][fix_id -
                #                                                        1] = localizer.localize(word_data["rawEEG"][0, i])
                # except:
                #     all_data[sentence_id -
                #              shard_start]["localized_eeg"][fix_id-1] = None
                res[0, sentence_id]["labels"][fix_id -
                                              1] = word_data["content"].item()

    for i in range(4):
        np.savez(os.path.join(
            out_dir, f"shard_{i}_{subject_name}.npz"), *(res[:, i*100:(i+1)*100]))
