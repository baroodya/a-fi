import torch
import time
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from movement_dataset import FeatureDataset
from model import NeuralNetwork
from constants import CURRENT_MODEL_PATH


training_dataset = FeatureDataset(
    "./movement_prediction/data/Top 15 Movement Data.csv"
)
test_dataset = FeatureDataset(
    "./movement_prediction/data/Test Data (IBM).csv"
)
training_batch_size = 1
validation_split = 0.1
shuffle_dataset = True
random_seed = 42
lr = 1e-4
epochs = 5
use_pretrained = False

# Creating data indices for training and validation splits:
training_dataset_size = len(training_dataset)
indices = list(range(training_dataset_size))
split = int(np.floor(validation_split * training_dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=training_batch_size,
    sampler=train_sampler,
)
validation_loader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=training_batch_size,
    sampler=valid_sampler,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
# define architecture
model = NeuralNetwork().to(device)
print(f"Model: {model}")
# for param in model.parameters():
#     print(param)

# define loss function and optimizer
loss_func = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

if not use_pretrained:
    for e in range(epochs):
        running_loss = 0
        # batch training
        batch_num = 1
        for features, labels in train_loader:
            # forward pass
            output = model(features)
            loss = loss_func(output, labels)
            if np.isnan(loss.item()):
                print(f"NaN reached. Breaking...\nFeatures: {features}")
                break

            if batch_num % 1000 == 0:
                progress = round(
                    (batch_num + (e * len(train_loader)))
                    / ((len(train_loader)) * epochs)
                    * 100,
                    2,
                )
                print(
                    f"Training on {training_dataset_size} datapoints. Progress: {progress}%. Output: {round(output.item(), 4)} Loss: {round(running_loss / batch_num,4)}",
                    end="\r",
                )
                running_loss = 0
            batch_num += 1

            # backward pass
            model.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    print()
    torch.save(
        model.state_dict(),
        CURRENT_MODEL_PATH,
    )
else:
    model.load_state_dict(torch.load(CURRENT_MODEL_PATH))

num_correct = 0
num_seen = 0
for features, label in validation_loader:
    output = model(features)
    loss = loss_func(output, label)

    if output > 0.5 and label == 1 or output < 0.5 and label == 0:
        num_correct += 1
    num_seen += 1

print(
    f"Validation done. Accuracy: {round(num_correct/num_seen * 100, 3)}%"
)

num_correct = 0
num_seen = 0
for features, label in test_loader:
    output = model(features)
    print(features)
    print(round(output.item(), 4))
    loss = loss_func(output, label)

    if output > 0.5 and label == 1 or output < 0.5 and label == 0:
        num_correct += 1
    num_seen += 1

print(
    f"Testing done. Accuracy: {round(num_correct/num_seen * 100, 3)}%"
)
