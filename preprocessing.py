import bs4 as bs
import pandas as pd
import numpy as np
import requests
import yfinance as yf

from constants import (SINGLE_TICKER_SYMBOL, STD_TICKER_SYMBOLS, FAANG_TICKER_SYMBOLS)


class DataPreprocessor():
    def __init__(self, ticker_symbol, validation_split) -> None:
        self.ticker_symbol = ticker_symbol
        self.validation_split = validation_split

    def pre_process_data(self):
        final_df = pd.DataFrame()
        try:
            stock = yf.Ticker(self.ticker_symbol)
            df = stock.history(period="3y")

            # Create Target
            df["Next Day Close"] = df["Close"].shift(-1)

            df = df.drop(columns=["Volume", "Dividends", "Stock Splits"])
            # # drop unnecessary columns
            # for c in df.columns:
            #     if np.std(df[c]) == 0.0:
            #         df = df.drop(columns=[c])

            df = df.dropna()

            final_df = pd.concat([final_df, df])
        except KeyError:
            print(f"Couldn't find {self.ticker_symbol}. Continuing...")

        val_start = int(len(final_df) * (1-self.validation_split))
        val_start_date = final_df.iloc[val_start].name

        self.train_df = final_df.loc[:val_start_date].copy()
        self.val_df = final_df.loc[val_start_date:].copy()
        
        # self.create_return_columns()
        self.target_column = "Next Day Close"
        self.feature_columns = self.train_df.columns.difference(
            [self.target_column])
        
    def get_train_df(self):
        return self.train_df

    def get_val_df(self):
        return self.val_df

    def get_dfs(self):
        return self.train_df, self.val_df

    def get_norm_dfs(self):
        return self.norm_train_df, self.norm_val_df

    def get_feature_columns(self):
        return self.feature_columns

    def get_target_column(self):
        return self.target_column
    
    def get_train_mean(self):
        return self.train_mean
    
    def get_train_stddev(self):
        return self.train_stddev

    def normalize_pre_processed_data(self, sequence_length):
        # normalize feature columns
        self.norm_train_df = self.train_df.copy()
        self.norm_val_df = self.val_df.copy()
        for c in self.train_df.columns:
            train_rolling = self.norm_train_df[c].rolling(sequence_length)
            self.norm_train_df[f"{c} Rolling Mean"] = train_rolling.mean()
            self.norm_train_df[f"{c} Rolling Std"] = train_rolling.std()

            val_rolling = self.norm_val_df[c].rolling(sequence_length)
            self.norm_val_df[f"{c} Rolling Mean"] = val_rolling.mean()
            self.norm_val_df[f"{c} Rolling Std"] = val_rolling.std()

            self.norm_train_df[c] = (
                self.norm_train_df[c] - self.norm_train_df[f"{c} Rolling Mean"]) / self.norm_train_df[f"{c} Rolling Std"]
            self.norm_val_df[c] = (self.norm_val_df[c] - self.norm_val_df[f"{c} Rolling Mean"]) / self.norm_val_df[f"{c} Rolling Std"]

        self.norm_train_df = self.norm_train_df.dropna()
        self.norm_val_df = self.norm_val_df.dropna()
        return self.norm_train_df, self.norm_val_df

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
