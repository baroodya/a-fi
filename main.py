import bs4 as bs
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import requests
import time
import re

from constants import (
    MOVEMENT_MODEL_PATH,
    PRICE_MODEL_PATH,
    TRAINING_WEIGHTS_FILE_NAME,
    VAL_WEIGHTS_FILE_NAME,
    TEST_WEIGHTS_FILE_NAME,
    STATS_FILE_NAME,
    SINGLE_TICKER_SYMBOL,
)
from preprocessing import pre_process_data
from movement_prediction.architectures import (
    MovementShallowRegressionLSTM
)
from price_prediction.architectures import (
    ShallowRegressionLSTM
)
from model import AFiMovementModel
from dataset import FeatureDataset
import parser

import torch
from torch.utils.data import DataLoader

# Collect Hyperparameters
args = parser.parse_args()

predict_movement = args.predict_movement[0]
training_batch_size = args.batch_size[0]
validation_split = args.val_split[0]
test_split = args.test_split[0]
shuffle_dataset = args.shuffle_dataset[0]
learning_rate = args.learning_rate[0]
epochs = args.epochs[0]
days_prior = args.days_prior[0]
use_pretrained = args.use_pretrained[0]
hidden_units = 16

architectures = []
if predict_movement:
    architectures.append(MovementShallowRegressionLSTM)
else:
    architectures.append(ShallowRegressionLSTM)

# num_ticker_symbols = args.num_ticker_symbols[0]

# # Get s&p 500 ticker symbols from wikipedia
# resp = requests.get(
#     "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# )
# soup = bs.BeautifulSoup(resp.text, "lxml")
# table = soup.find("table", {"class": "wikitable sortable"})

# ticker_symbols = []

# for row in table.findAll("tr")[1:num_ticker_symbols + 1]:
#     ticker = row.findAll("td")[0].text.strip()
#     ticker_symbols.append(ticker)

ticker_symbols = SINGLE_TICKER_SYMBOL

# Create the dataset
training_df, val_df, test_df, feature_columns, target_columns = pre_process_data(
    ticker_symbols=ticker_symbols, validation_split=validation_split, test_split=test_split
)

num_features = len(feature_columns)
target_idx = 1
if predict_movement:
    target_idx = 0

training_dataset = FeatureDataset(
    dataframe=training_df, features=feature_columns, target=target_columns[target_idx], sequence_length=days_prior)

val_dataset = FeatureDataset(
    dataframe=val_df, features=feature_columns, target=target_columns[target_idx], sequence_length=days_prior)

test_dataset = FeatureDataset(
    dataframe=test_df, features=feature_columns, target=target_columns[target_idx], sequence_length=days_prior)

# for repeatability
torch.manual_seed(99)

training_loader = DataLoader(
    training_dataset, batch_size=training_batch_size, shuffle=shuffle_dataset)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=shuffle_dataset)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=shuffle_dataset)

# Model , Optimizer, Loss
model = architectures[0](
    num_sensors=num_features, hidden_units=hidden_units)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model.to(device)

loss_fn = torch.nn.MSELoss()
if predict_movement:
    loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

framework = AFiMovementModel(
    model=model, loss_function=loss_fn, optimizer=optimizer)

losses = framework.train(train_loader=training_loader, epochs=epochs)

train_data = framework.test(training_loader, predict_movement=predict_movement)
print(
    f"Training done. Accuracy: {round(train_data['accuracy'] * 100, 3)}%"
)

validation_data = framework.test(val_loader, predict_movement=predict_movement)
print(
    f"Validation done. Accuracy: {round(validation_data['accuracy'] * 100, 3)}%"
)

test_data = framework.test(
    test_loader, predict_movement=predict_movement, threshold=1)
print(
    f"Testing done. Accuracy: {round(test_data['accuracy'] * 100, 3)}%"
)

current_model_path = PRICE_MODEL_PATH
if predict_movement:
    current_model_path = MOVEMENT_MODEL_PATH

best_training_acc = 0.0
best_val_acc = 0.0
best_test_acc = 0.0

with open(current_model_path + STATS_FILE_NAME, 'r+') as f:
    lines = f.readlines()

    best_training_acc = float(re.findall("\d+\.\d+", lines[0])[0])
    best_val_acc = float(re.findall("\d+\.\d+", lines[1])[0])
    best_test_acc = float(re.findall("\d+\.\d+", lines[2])[0])

    if train_data['accuracy'] > best_training_acc:
        torch.save(
            model.state_dict(),
            current_model_path + TRAINING_WEIGHTS_FILE_NAME,
        )
        best_training_acc = train_data["accuracy"]

    if validation_data['accuracy'] > best_val_acc:
        torch.save(
            model.state_dict(),
            current_model_path + VAL_WEIGHTS_FILE_NAME,
        )
        best_val_acc = validation_data["accuracy"]

    if test_data['accuracy'] > best_test_acc:
        torch.save(
            model.state_dict(),
            current_model_path + TEST_WEIGHTS_FILE_NAME,
        )
        best_test_acc = test_data["accuracy"]
    f.seek(0)
    f.truncate()
    f.write(
        f"training: {best_training_acc * 100:.2f}%\nvalidation: {best_val_acc * 100:.2f}%\ntest: {best_test_acc * 100:.2f}%")