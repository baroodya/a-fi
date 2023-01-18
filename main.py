import csv

import torch
import time
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import FeatureDataset
from model import NeuralNetwork

dataset = FeatureDataset("./data/Google Data.csv")
batch_size = 1
validation_split = 0.2
shuffle_dataset = True
random_seed = 4
lr = 1e-15
epochs = 5
epsilon = 1

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=train_sampler
)
validation_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, sampler=valid_sampler
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
# define architecture
model = NeuralNetwork().to(device)
print(model)
print(model.parameters)

# define loss function and optimizer
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for e in range(epochs):
    running_loss = 0
    # batch training
    batch_num = 1
    for features, labels in train_loader:
        # forward pass
        output = model(features)
        print(f"Output: {output}", end="\r")
        time.sleep(0.1)
        loss = loss_func(output, labels)

        # backward pass
        model.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
print()
num_correct = 0
num_seen = 0
for features, label in validation_loader:
    output = model(features)
    print(abs(output - label))
    loss = loss_func(output, label)

    if abs(output - label) < epsilon:
        num_correct += 1
    num_seen += 1

print(f"Testing done. Accuracy: {round(num_correct/num_seen * 100, 3)}")
