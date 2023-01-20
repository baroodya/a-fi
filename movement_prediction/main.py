import torch
import time
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from movement_dataset import MovementFeatureDataset
from model import AFiMovementModel
from constants import (
    CURRENT_MODEL_PATH,
    TRAIN_TICKER_SYMBOLS,
    TEST_TICKER_SYMBOLS,
)


training_dataset = MovementFeatureDataset(TRAIN_TICKER_SYMBOLS)
test_dataset = MovementFeatureDataset(TEST_TICKER_SYMBOLS)
training_batch_size = 1
validation_split = 0.1
shuffle_dataset = True
random_seed = 42
lr = 1e-3
epochs = 5
use_pretrained = False
num_features = training_dataset.X.shape[1]

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

loss_func = torch.nn.BCELoss()
model = AFiMovementModel(num_features, loss_func).to(device)
print(f"Model: {model}")

# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if not use_pretrained:
    model.train(train_loader, optimizer, epochs)
    torch.save(
        model.state_dict(),
        CURRENT_MODEL_PATH,
    )
else:
    model.load_state_dict(torch.load(CURRENT_MODEL_PATH))

validation_data = model.test(validation_loader)
print(
    f"Validation done. Accuracy: {round(validation_data['accuracy'] * 100, 3)}%"
)

test_data = model.test(test_loader)
print(
    f"Testing done. Accuracy: {round(test_data['accuracy'] * 100, 3)}%"
)
