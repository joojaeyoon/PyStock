import os

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

import pandas as pd
import numpy as np


class StockDataset(Dataset):
    def __init__(self, Datadir):
        self.filelist = [f for f in os.listdir(Datadir) if f.endswith("csv")]
        self.Data = torch.zeros(len(self.filelist), 200, 5)

        for i, file in enumerate(self.filelist):
            n = pd.read_csv(file).to_numpy()[:, 1:]
            tensor = torch.FloatTensor(np.array(n, dtype=np.float32))
            self.Data[i] = tensor

    def __getitem__(self, index):
        return self.Data[index]

    def __len__(self):
        return len(self.Data)


t = StockDataset("./")
