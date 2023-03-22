import bs4 as bs
import pandas as pd
import numpy as np
import requests
import yfinance as yf

from constants import (SINGLE_TICKER_SYMBOL, STD_TICKER_SYMBOLS, FAANG_TICKER_SYMBOLS)


class DataPreprocessor():
    def __init__(self, ticker_symbol, validation_split, test_split) -> None:
        self.ticker_symbol = ticker_symbol
        self.validation_split = validation_split
        self.test_split = test_split

    def pre_process_data(self):
        final_df = pd.DataFrame()
        try:
            stock = yf.Ticker(self.ticker_symbol)
            df = stock.history(period="5y")

            # Create Labels
            df["Next Day Movement"] = df["Close"] < df[
                "Close"
            ].shift(-1)

            df["Next Day Close"] = df["Close"].shift(-1)

            # trim NaNs
            df = df.dropna()
            droppables = ["Volume", "Dividends", "Stock Splits"]
            for droppable in droppables:
                if np.mean(df[droppable] != 0.0):
                    droppables.remove(droppable)

            df = df.drop(columns=droppables)
            # convert types
            df = df.astype({"Next Day Movement": "int"})

            final_df = pd.concat([final_df, df])
        except KeyError:
            print(f"Couldn't find {self.ticker_symbol}. Continuing...")

        test_start = int(len(final_df) * (1-self.test_split))
        test_start_date = final_df.iloc[test_start].name

        val_start = int(test_start - (len(final_df) * self.validation_split))
        val_start_date = final_df.iloc[val_start].name

        self.train_df = final_df.loc[:val_start_date].copy()
        self.val_df = final_df.loc[val_start_date:test_start_date].copy()
        self.test_df = final_df.loc[test_start_date:].copy()
    def get_train_df(self):
        return self.train_df

    def get_val_df(self):
        return self.val_df

    def get_test_df(self):
        return self.test_df

    def get_dfs(self):
        return self.train_df, self.val_df, self.test_df

    def get_norm_dfs(self):
        return self.norm_train_df, self.norm_val_df, self.norm_test_df

    def get_feature_columns(self):
        return self.feature_columns

    def get_target_columns(self):
        return self.target_columns

    def normalize_pre_processed_data(self):
        # normalize feature columns
        self.target_columns = ["Next Day Movement", "Next Day Close"]
        self.feature_columns = self.train_df.columns.difference(
            self.target_columns)


        self.norm_train_df = self.train_df.copy()
        self.norm_val_df = self.val_df.copy()
        self.norm_test_df = self.test_df.copy()
        for c in self.feature_columns:
            train_mean = self.train_df[c].mean()
            train_stddev = self.train_df[c].std()

            self.norm_train_df[c] = (
                self.train_df[c] - train_mean) / train_stddev
            self.norm_val_df[c] = (self.val_df[c] - train_mean) / train_stddev
            self.norm_test_df[c] = (
                self.test_df[c] - train_mean) / train_stddev


def get_ticker_symbols(num_ticker_symbols):
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

    ticker_symbols = FAANG_TICKER_SYMBOLS
    ticker_symbols = SINGLE_TICKER_SYMBOL
    return ticker_symbols
