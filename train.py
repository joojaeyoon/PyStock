import torch
import torch.nn as nn
import torch.optim as optim


from .models import LSTMModel
from .dataset import StockDataset

input_dim = 5
hidden_dim = 100
layer_dim = 1
output_dim = 10

num_epochs = 10

dset = StockDataset("./")

train_loader = torch.utils.data.DataLoader(dataset=dset)

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

criterion = nn.CrossEntropyLoss()

learning_late = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_late)

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):

        optimizer.zero_grad()

        outputs = model(data)

        loss = criterion(outputs, "Something")

        loss.backward()

        optimizer
