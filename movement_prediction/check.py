import torch
import time
import numpy as np
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from movement_dataset import MovementFeatureDataset
from model import AFiMovementModel
from constants import CURRENT_MODEL_PATH

starting_value = 100
starting_shares = 0

file_out = pd.read_csv("./movement_prediction/data/Test Data (IBM).csv")
feature_cols = file_out.columns.difference(["Next Day Movement"])
num_features = file_out.shape[1] - 1

x = file_out.iloc[:-1, :num_features].to_numpy(dtype=float)
y = file_out.iloc[:-1, num_features].values

test_dataset = MovementFeatureDataset(
    "./movement_prediction/data/Test Data (IBM).csv"
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
)

model = AFiMovementModel()
print(f"Model: {model}")
model.load_state_dict(torch.load(CURRENT_MODEL_PATH))

curr_value = starting_value
curr_shares = starting_shares
i = 0
for feat, _ in test_loader:
    day_data = x[i]
    close = day_data[4]

    output = model(feat)
    print(feat)
    if output > 0.5:
        curr_shares += (output) * (curr_value / close)
        curr_value = 0
    else:
        curr_value += (1 - output) * (curr_shares * close)
        curr_shares = 0
    i += 1
curr_value += curr_shares * x[-1][4]

print(curr_value)
