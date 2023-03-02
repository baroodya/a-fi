import bs4 as bs
import matplotlib.pyplot as plt
import numpy as np
import requests
import time

from constants import (
    CURRENT_MODEL_PATH,
    TRAIN_TICKER_SYMBOLS,
    TEST_TICKER_SYMBOLS,
)
from model import AFiMovementModel
from movement_dataset import MovementFeatureDataset
import parser

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Collect Hyperparameters
args = parser.parse_args()

training_batch_size = args.batch_size[0]
validation_split = args.val_split[0]
shuffle_dataset = args.shuffle_dataset[0]
learning_rate = args.learning_rate[0]
epochs = args.epochs[0]
days_prior = args.days_prior[0]
use_pretrained = args.use_pretrained[0]

# Get s&p 500 ticker symbols from wikipedia
resp = requests.get(
    "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
)
soup = bs.BeautifulSoup(resp.text, "lxml")
table = soup.find("table", {"class": "wikitable sortable"})

train_ticker_symbols = []

for row in table.findAll("tr")[1:]:
    ticker = row.findAll("td")[0].text.strip()
    train_ticker_symbols.append(ticker)

# Create the datasets
training_dataset = MovementFeatureDataset(
    ticker_symbols=train_ticker_symbols, len_history=days_prior
)
test_dataset = MovementFeatureDataset(
    ticker_symbols=TEST_TICKER_SYMBOLS, len_history=days_prior
)

num_features = training_dataset.X.shape[1]

# Creating data indices for training and validation splits:
training_dataset_size = len(training_dataset)
indices = list(range(training_dataset_size))
split = int(np.floor(validation_split * training_dataset_size))
if shuffle_dataset:
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(
    training_dataset,
    batch_size=training_batch_size,
    sampler=train_sampler,
)
train_acc_loader = DataLoader(
    training_dataset,
    batch_size=1,
    sampler=train_sampler,
)
validation_loader = DataLoader(
    training_dataset,
    batch_size=1,
    sampler=valid_sampler,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
)

# Model , Optimizer, Loss
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

loss_fn = torch.nn.BCELoss()
model = AFiMovementModel(num_features, loss_fn).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if not use_pretrained:
    losses = model.train(train_loader, optimizer, epochs)
    # plt.plot(losses)
    # plt.title("Loss during Training")
    # plt.xlabel("# Datapoints Seen")
    # plt.ylabel("loss")
    torch.save(
        model.state_dict(),
        CURRENT_MODEL_PATH,
    )
else:
    model.load_state_dict(torch.load(CURRENT_MODEL_PATH))

train_data = model.test(train_acc_loader)
print(
    f"Training done. Accuracy: {round(train_data['accuracy'] * 100, 3)}%"
)

validation_data = model.test(validation_loader)
print(
    f"Validation done. Accuracy: {round(validation_data['accuracy'] * 100, 3)}%"
)

test_data = model.test(test_loader)
print(
    f"Testing done. Accuracy: {round(test_data['accuracy'] * 100, 3)}%"
)
