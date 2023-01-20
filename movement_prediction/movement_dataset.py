import pandas as pd
import numpy as np
import yfinance as yf
import torch
from torch.utils.data import Dataset
from constants import TICKER_SYMBOLS


class MovementFeatureDataset(Dataset):
    def __init__(self, file_name, scale=True):

        df = pd.DataFrame()
        for symbol in TICKER_SYMBOLS:
            stock = yf.Ticker(symbol)
            df = pd.concat(
                [df, stock.history(period="max").reset_index()]
            )

            # Create Labels
            df["Next Day Movement"] = df["Close"] < df["Close"].shift(
                -1
            )

            # trim NaNs
            df = df.dropna()

            # convert types
            df = df.astype({"Next Day Movement": "int"})
        df = df.drop(columns=["Dividends"])

        # normalize feature columns
        feature_cols = df.columns.difference(["Next Day Movement"])
        normalized_df = df[feature_cols].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )
        num_features = df.shape[1] - 1

        x = normalized_df.iloc[:-1, :num_features].to_numpy(dtype=float)
        y = df.iloc[:-1, num_features].values

        # Converting to torch tensors
        self.X = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
