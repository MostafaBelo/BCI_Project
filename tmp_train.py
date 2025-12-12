# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from torcheeg.models import EEGNet

from tqdm import tqdm

from EEGDataset import EEGDataset, WordEEGDataset

# %%
MODEL_NAME = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# %%
# clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
# clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device)

model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

# %%
embeddings = model.encode(["hello world", "open source embeddings"])
embeddings.shape, embeddings.dtype, type(embeddings)

# %%
model

# %%
# ds = EEGDataset("shards", pad_upto=6000)
# # ds = EEGDataset("/home/mostafaelfaggal/Documents/BCI", pad_upto=6000)
# ds[0][0].shape, ds[0][1]

# %%
# train_ds = WordEEGDataset("shards", pad_upto=200, selective_indexing=list(range(274)))
# val_ds = WordEEGDataset("shards", pad_upto=200, selective_indexing=list(range(274,332)))
# test_ds = WordEEGDataset("shards", pad_upto=200, selective_indexing=list(range(332,392)))

train_ds = WordEEGDataset("shards", pad_upto=200,
                          selective_indexing=list(range(40)))
val_ds = WordEEGDataset("shards", pad_upto=200,
                        selective_indexing=list(range(40, 45)))
test_ds = WordEEGDataset("shards", pad_upto=200,
                         selective_indexing=list(range(45, 49)))

# %%
# train_ds, val_ds, test_ds = ds.split_train_valid_test(train_ratio=0.7, valid_ratio=0.15, shuffle=False)

# train_dl = train_ds.getLoader(batch_size=1, num_workers=0)
# val_dl = val_ds.getLoader(batch_size=1, num_workers=0)
# test_dl = test_ds.getLoader(batch_size=1, num_workers=0)

# len(train_ds), len(val_ds), len(test_ds)

# %%
train_ds[0][0].shape, len(train_ds[0][1]), train_ds[0][1][0]

# %%
val_ds[0][0].shape, len(val_ds[0][1]), val_ds[0][1][0]

# %%
# embeddings = model.encode([ds[0][1]])
embeddings = model.encode([train_ds[0][1]])
embeddings.shape, embeddings.dtype, type(embeddings)

# %%
# ds[0][0].shape

# for batch_data, batch_labels in train_dl:
#     print(len(batch_data))
#     print(len(batch_labels))
#     break

# %%


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.text_encoder_model = SentenceTransformer("all-MiniLM-L6-v2")

    def forward(self, texts):
        embeddings = self.text_encoder_model.encode(
            texts, convert_to_tensor=True)
        return embeddings


class LocalizedEEGEncoder(nn.Module):
    def __init__(self, ch_count=8196, embedding_dim=384):
        super(LocalizedEEGEncoder, self).__init__()

        self.temporal = nn.Sequential(
            nn.Conv1d(ch_count, 1024, 11, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(1024, 512, 11, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(512, 256, 11, padding=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((256, 1))
        )

        self.fc = nn.Linear(256, embedding_dim)

        # self.eeg_encoder = EEGNet(chunk_size=10000,
        #                     num_electrodes=ch_count,
        #                     dropout=0.3,
        #                     kernel_1=64,
        #                     kernel_2=16,
        #                     F1=8,
        #                     F2=16,
        #                     D=2,
        #                     num_classes=embedding_dim)

    def forward(self, x):
        # x = torch.fft.rfft(x, dim=2)
        # x = torch.log(torch.abs(x) + 1e-8)
        x = self.temporal(x).squeeze(-1)
        x = self.fc(x)

        # x = self.eeg_encoder(x)

        x = F.normalize(x, p=2, dim=1)
        return x


class EEGCLIPModel(nn.Module):
    def __init__(self, ch_count=8196, embedding_dim=384, freeze_text=True):
        super(EEGCLIPModel, self).__init__()
        self.text_encoder = TextEncoder()
        self.eeg_encoder = LocalizedEEGEncoder(
            ch_count=ch_count, embedding_dim=embedding_dim)

        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, eeg_data, texts):
        eeg_embeddings = self.eeg_encoder(eeg_data)
        text_embeddings = self.text_encoder(texts)
        return eeg_embeddings, text_embeddings

# %% [markdown]
# # Training


# %%
model = EEGCLIPModel().to(device)
# model = EEGCLIPModel(3438).to(device)
# model.load_state_dict(torch.load("best_model.pt"))

# %%
# def train(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, epochs: int = 10):


def train(model: nn.Module, train_dataset: WordEEGDataset, valid_dataset: WordEEGDataset, epochs: int = 10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_valid_loss = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_count = 0
        # for batch in tqdm(train_loader):
        for batch in tqdm(train_dataset):
            batch = ([batch[0]], [batch[1]])
            for i in range(len(batch[0])):
                eeg_data = batch[0][i].to(torch.float32).to(device)
                texts = batch[1][i]

                optimizer.zero_grad()
                eeg_embeddings, text_embeddings = model(eeg_data, texts)

                # loss = ((eeg_embeddings - text_embeddings) ** 2).mean()
                loss = 1.0 - \
                    F.cosine_similarity(eeg_embeddings, text_embeddings).mean()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                train_count += 1

        # avg_loss = total_loss / len(train_loader)
        avg_loss = total_loss / train_count

        model.eval()
        total_valid_loss = 0.0
        valid_count = 0
        with torch.inference_mode():
            # for batch in tqdm(valid_loader):
            for batch in tqdm(valid_dataset):
                batch = ([batch[0]], [batch[1]])
                for i in range(len(batch[0])):
                    eeg_data = batch[0][i].to(torch.float32).to(device)
                    texts = batch[1][i]

                    eeg_embeddings, text_embeddings = model(eeg_data, texts)

                    # loss = ((eeg_embeddings - text_embeddings) ** 2).mean()
                    loss = 1.0 - \
                        F.cosine_similarity(
                            eeg_embeddings, text_embeddings).mean()

                    total_valid_loss += loss.item()
                    valid_count += 1

        # avg_valid_loss = total_valid_loss / len(valid_loader)
        avg_valid_loss = total_valid_loss / valid_count

        if (best_valid_loss is None) or (avg_valid_loss < best_valid_loss):
            print(f"Valid Loss: {avg_valid_loss:.10f}")
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), "best_model.pt")

        print(
            f"Epoch [{epoch+1}/{epochs}]:- Train Loss: {avg_loss:.6f} | Valid Loss: {avg_valid_loss:.6f}")
        torch.save(model.state_dict(), "last_model.pt")

        torch.cuda.empty_cache()


# %%
# def test(model: nn.Module, test_loader: DataLoader):
def test(model: nn.Module, test_dataset: WordEEGDataset):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.inference_mode():
        # for batch in tqdm(test_loader):
        for batch in tqdm(test_dataset):
            batch = ([batch[0]], [batch[1]])
            for i in range(len(batch[0])):
                eeg_data = batch[0][i].to(torch.float32).to(device)
                texts = batch[1][i]

                eeg_embeddings, text_embeddings = model(eeg_data, texts)

                # loss = ((eeg_embeddings - text_embeddings) ** 2).mean()
                loss = 1.0 - \
                    F.cosine_similarity(eeg_embeddings, text_embeddings).mean()

                total_loss += loss.item()
                count += 1

    avg_loss = total_loss / count
    print(f"Test Loss: {avg_loss:.6f}")


# %%
# train(model, train_dl, val_dl, epochs=20)
train(model, train_ds, val_ds, epochs=3)
# train(model, ds, epochs=20)

# %%
# test(model, test_dl)
test(model, test_ds)
# test(model, ds)
