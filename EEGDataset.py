import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import os

import time

current_shard_idx = -1
current_loaded_shard = None


class FileManager:
    def __init__(self):
        self.state = "idle"

    def get_shard(self, shard_idx, shard_path):
        if (self.state == "pulling"):
            while True:
                if (self.state != "idle"):
                    time.sleep(.1)

        if current_shard_idx != shard_idx:
            self._load_shard(shard_idx, shard_path)

    def _load_shard(self, shard_idx, shard_path):
        global current_loaded_shard, current_shard_idx
        self.state = "pulling"
        current_loaded_shard = np.load(
            shard_path, allow_pickle=True)["arr_0"]
        current_shard_idx = shard_idx
        self.state = "idle"


manager = FileManager()


class EEGDataset(Dataset):
    def __init__(self, data_dir, shard_size=100, pad_upto=10000, crp_rng=(0, 1), selective_indexing=None):
        """
        Args:
            data_dir (string): Path to the shards folder with .npy files with EEG data.
        """
        super(EEGDataset, self).__init__()

        self.data_dir = data_dir
        self.shard_size = shard_size

        self.padding = pad_upto
        self.crp_rng = crp_rng

        self.selective_indexing = selective_indexing

        if not (os.path.exists(os.path.join(data_dir, "labels.txt"))):
            raise FileNotFoundError(
                "Labels file not found in the specified directory.")
        with open(os.path.join(data_dir, "labels.txt"), 'r') as f:
            self.labels = [line.split(":", 2)
                           for line in f.read().strip().split("\n")]
            self.labels = [(int(idx), sentiment, sentence)
                           for idx, sentiment, sentence in self.labels]

        self.subject_dirs = []
        for file in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, file)):
                self.subject_dirs.append(os.path.join(data_dir, file))

        self.indexing = []
        for subject_i in range(len(self.subject_dirs)):
            for label_i in range(len(self.labels)):
                self.indexing.append((subject_i, label_i))

    def __len__(self):
        if self.selective_indexing is not None:
            return len(self.selective_indexing)
        return len(self.indexing)

    def __getitem__(self, idx: int):
        global current_shard_idx, current_loaded_shard
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")

        # if (self.selective_indexing is not None) and (idx not in self.selective_indexing):
        #     raise IndexError("Index not in selective indexing list")
        if self.selective_indexing is not None:
            idx = self.selective_indexing[idx]

        subj_i, lbl_i = self.indexing[idx]
        _, sent, lbl = self.labels[lbl_i]

        shard_idx = lbl_i // self.shard_size
        shard_path = os.path.join(
            self.subject_dirs[subj_i], f"shard_{shard_idx}.npz")
        shard_idx = subj_i*4 + shard_idx
        if shard_idx != current_shard_idx:
            manager.get_shard(shard_idx, shard_path)

        data = current_loaded_shard[lbl_i % self.shard_size]
        if data is None:
            return None, sent, lbl

        l = data.shape[1]
        data = data[:, int(self.crp_rng[0]*l):int(self.crp_rng[1]*l)]
        pad_width = self.padding - data.shape[1]
        if pad_width > 0:
            data = np.concatenate(
                [data, np.zeros((data.shape[0], pad_width))], axis=1)
        else:
            data = data[:, :self.padding]
        return torch.tensor(data), sent, lbl

    def _collate_fn(self, batch):
        try:
            data = [item[0] for item in batch if item[0] is not None]
            data = torch.stack(data, dim=0)
            sent = [int(item[1])+1 for item in batch if item[0] is not None]
            sent = torch.tensor(sent)
            labels = [item[2] for item in batch if item[0] is not None]
            return data, sent, labels
        except:
            return None, None, None

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
            self.data_dir, self.shard_size, self.padding, self.crp_rng, selective_indexing=train_indices)
        valid_dataset = EEGDataset(
            self.data_dir, self.shard_size, self.padding, self.crp_rng, selective_indexing=valid_indices)
        test_dataset = EEGDataset(
            self.data_dir, self.shard_size, self.padding, self.crp_rng, selective_indexing=test_indices)

        return train_dataset, valid_dataset, test_dataset

    def getLoader(self, batch_size: int, num_workers: int = 0):
        if num_workers == 0:
            return DataLoader(self, batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn=self._collate_fn)
        else:
            return DataLoader(self, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2, collate_fn=self._collate_fn)


class WordEEGDataset(Dataset):
    def __init__(self, data_dir, shard_size=50, pad_upto=10000, selective_indexing=None):
        """
        Args:
            data_dir (string): Path to the shards folder with .npy files with EEG data.
        """
        super(WordEEGDataset, self).__init__()

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

        idx, _ = self.labels[idx]

        shard_idx = idx // self.shard_size
        if shard_idx != current_shard_idx:
            shard_path = os.path.join(
                self.data_dir, f"words/shard_{shard_idx}.npy")
            current_loaded_shard = np.load(shard_path, allow_pickle=True)
            current_shard_idx = shard_idx

        data = current_loaded_shard[idx % self.shard_size]
        data["eeg"] = list(data["eeg"])
        data["localized_eeg"] = list(data["localized_eeg"])
        data["labels"] = list(data["labels"])
        for i in range(len(data["localized_eeg"])):
            if data["localized_eeg"][i] is None:
                continue
            eeg_shape = data["localized_eeg"][i].shape
            if eeg_shape[1] < self.padding:
                pad_width = self.padding - \
                    eeg_shape[1]
                if pad_width > 0:
                    data["localized_eeg"][i] = np.concatenate(
                        [data["localized_eeg"][i], np.zeros((eeg_shape[0], pad_width))], axis=1)
            elif eeg_shape[1] > self.padding:
                data["localized_eeg"][i] = data["localized_eeg"][i][:, :self.padding]

            if not isinstance(data["localized_eeg"][i], torch.Tensor):
                data["localized_eeg"][i] = torch.tensor(
                    data["localized_eeg"][i])

        signal = [data["localized_eeg"][i] for i in range(
            len(data["localized_eeg"])) if (data["localized_eeg"][i] is not None)]
        labels = [data["labels"][i] for i in range(
            len(data["localized_eeg"])) if (data["localized_eeg"][i] is not None)]
        return torch.stack(signal, dim=0), labels

    def _collate_fn(self, batch):
        data = [item[0] for item in batch if item[0] is not None]
        # data = torch.stack(data, dim=0)
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

        train_dataset = WordEEGDataset(
            self.data_dir, self.shard_size, selective_indexing=train_indices)
        valid_dataset = WordEEGDataset(
            self.data_dir, self.shard_size, selective_indexing=valid_indices)
        test_dataset = WordEEGDataset(
            self.data_dir, self.shard_size, selective_indexing=test_indices)

        return train_dataset, valid_dataset, test_dataset

    def getLoader(self, batch_size: int, num_workers: int = 0):
        if num_workers == 0:
            return DataLoader(self, batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn=self._collate_fn)
        else:
            return DataLoader(self, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2, collate_fn=self._collate_fn)
