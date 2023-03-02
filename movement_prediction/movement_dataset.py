import pandas as pd
import torch
import yfinance as yf


class MovementFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, ticker_symbols, len_history):
        final_df = pd.DataFrame()
        for i, symbol in enumerate(ticker_symbols):
            try:
                stock = yf.Ticker(symbol)
                df = stock.history(period="max").reset_index()

                for j in range(len_history):
                    df[f"{j+1} Day(s) Ago Close"] = df["Close"].shift(
                        j + 1
                    )
                    df[f"{j+1} Day(s) Ago Open"] = df["Open"].shift(
                        j + 1
                    )
                    df[f"{j+1} Day(s) Ago Volume"] = df["Volume"].shift(
                        j + 1
                    )

                # Create Labels
                df["Next Day Movement"] = df["Close"] < df[
                    "Close"
                ].shift(-1)

                # trim NaNs
                df = df.dropna()

                # convert types
                df = df.astype({"Next Day Movement": "int"})

                # drop useless columns
                df = df.drop(columns=["Dividends"])

                final_df = pd.concat([final_df, df])
            except KeyError:
                print(f"Couldn't find {symbol}. Continuing...")
            print(
                f"Collecting the most recent financial data. Progress: {round((i+1)/len(ticker_symbols) *100, 2)}%.",
                end="\r",
            )
        print("\nDone!")
        print(final_df)
        # normalize feature columns
        feature_cols = final_df.columns.difference(
            ["Next Day Movement"]
        )
        normalized_df = final_df[feature_cols].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )
        num_features = final_df.shape[1] - 1

        x = normalized_df.iloc[:-1, :num_features].to_numpy(dtype=float)
        y = final_df.iloc[:-1, num_features].values

        # Converting to torch tensors
        self.X = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
