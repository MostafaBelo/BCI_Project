import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import os

current_shard_idx = -1
current_loaded_shard = None


class EEGDataset(Dataset):
    def __init__(self, data_dir, shard_size=50, pad_upto=10000, selective_indexing=None):
        """
        Args:
            data_dir (string): Path to the shards folder with .npy files with EEG data.
        """
        super(EEGDataset, self).__init__()

        self.data_dir = data_dir
        self.shard_size = shard_size

        self.padding = pad_upto

        self.selective_indexing = selective_indexing

        if not (os.path.exists(os.path.join(data_dir, "labels.txt"))):
            raise FileNotFoundError(
                "Labels file not found in the specified directory.")
        with open(os.path.join(data_dir, "labels.txt"), 'r') as f:
            self.labels = [line.split(":", 1)
                           for line in f.read().strip().split("\n")]
            self.labels = [(int(idx), sent) for idx, sent in self.labels]

    def __len__(self):
        if self.selective_indexing is not None:
            return len(self.selective_indexing)
        return len(self.labels)

    def __getitem__(self, idx: int):
        global current_shard_idx, current_loaded_shard
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")

        # if (self.selective_indexing is not None) and (idx not in self.selective_indexing):
        #     raise IndexError("Index not in selective indexing list")
        if self.selective_indexing is not None:
            idx = self.selective_indexing[idx]

        idx, lbl = self.labels[idx]

        shard_idx = idx // self.shard_size
        if shard_idx != current_shard_idx:
            shard_path = os.path.join(
                self.data_dir, f"localized_eeg_sentences_ZAB_shard{shard_idx}.npy")
            current_loaded_shard = np.load(shard_path, allow_pickle=True)
            current_shard_idx = shard_idx

        data = current_loaded_shard[idx % self.shard_size]
        if data is None:
            return None, lbl

        if data.shape[1] < self.padding:
            pad_width = self.padding - data.shape[1]
            if pad_width > 0:
                data = np.concatenate(
                    [data, np.zeros((data.shape[0], pad_width))], axis=1)
            else:
                data = data[:, :self.padding]
        return torch.tensor(data), lbl

    def _collate_fn(self, batch):
        data = [item[0] for item in batch if item[0] is not None]
        data = torch.stack(data, dim=0)
        labels = [item[1] for item in batch if item[0] is not None]
        return data, labels

    def split_train_valid_test(self, train_ratio=0.8, valid_ratio=0.1, shuffle=False):
        total_size = len(self)
        train_size = int(total_size * train_ratio)
        valid_size = int(total_size * valid_ratio)
        test_size = total_size - train_size - valid_size

        indices = list(range(total_size))
        if shuffle:
            np.random.seed(42)
            np.random.shuffle(indices)

        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size + valid_size]
        test_indices = indices[train_size + valid_size:]

        train_dataset = EEGDataset(
            self.data_dir, self.shard_size, selective_indexing=train_indices)
        valid_dataset = EEGDataset(
            self.data_dir, self.shard_size, selective_indexing=valid_indices)
        test_dataset = EEGDataset(
            self.data_dir, self.shard_size, selective_indexing=test_indices)

        return train_dataset, valid_dataset, test_dataset

    def getLoader(self, batch_size: int, num_workers: int = 0):
        if num_workers == 0:
            return DataLoader(self, batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn=self._collate_fn)
        else:
            return DataLoader(self, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2, collate_fn=self._collate_fn)
