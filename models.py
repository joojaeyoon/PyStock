import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

import os
import numpy as np
import pandas as pd


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(
            0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(
            0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1:])

        return out


if __name__ == "__main__":

    test = LSTMModel(28, 100, 1, 10)
    print(test.forward(torch.zeros(28, 1, 28)).shape)
