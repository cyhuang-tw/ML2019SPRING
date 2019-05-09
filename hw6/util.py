import os
import sys
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def read_file(data_file, label_file=None):
    df = pd.read_csv(data_file)
    text = df.values[:, 1].reshape(-1, 1)

    if label_file == None:
        return text.flatten()
    else:
        df = pd.read_csv(label_file)
        label = df.values[:, 1].reshape(-1, 1)
        data = np.concatenate((text, label), axis=1)
        #np.random.seed(666)
        np.random.shuffle(data)
        val_split = 0.2
        index = np.floor((1 - val_split) * data.shape[0]).astype(np.int)
        text = data[:, 0]
        label = data[:, 1]
        train_text, val_text = np.split(text, [index])
        train_label, val_label = np.split(label, [index])
        return train_text, train_label, val_text, val_label

def collate_fn(data):
    text, label, length = map(list, zip(*data))
    number = [x for x in range(len(text))]
    length, text, label, number = map(list, zip(*sorted(zip(length, text, label, number), reverse=True)))
    label = torch.tensor(label, dtype=torch.long)
    length = torch.tensor(length, dtype=torch.long)

    return text, label, length, number

class TextDataset(Dataset):
    def __init__(self, data, label, max_length):
        self.data = data
        self.label = label
        self.max_length = max_length

        self.length = [min(self.max_length, len(text)) for text in self.data]

        for i, text in enumerate(self.data):
            if len(text) > self.max_length:
                self.data[i] = text[:self.max_length]
            else:
                self.data[i] = self.data[i] + ['<PAD>'] * (self.max_length - len(text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.length[index]

class TextNet(nn.Module):
    def __init__(self, weights, max_length):
        super(TextNet, self).__init__()

        self.embedding_dim = 256
        self.hidden_dim = 256
        self.num_layers = 2

        self.max_length = max_length

        self.embedding = nn.Embedding.from_pretrained(weights, freeze=False)

        self.LSTM = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.5
            )

        self.fc1 = nn.Sequential(
            nn.Linear(self.max_length * self.hidden_dim * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5)
            )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
            )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5)
            #nn.Softmax(dim=1)
            )
        self.fc4 = nn.Sequential(
            nn.Linear(64, 2)
            )

    def forward(self, batch, length):
        input_embedding = self.embedding(batch)

        output, (ht, ct) = self.LSTM(input_embedding)
        
        x = output.contiguous()
        x = x.view(-1, self.max_length * self.hidden_dim * 2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x