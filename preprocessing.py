import bs4 as bs
import pandas as pd
import requests
import yfinance as yf

from constants import SINGLE_TICKER_SYMBOL


def pre_process_data(num_ticker_symbols, validation_split, test_split):
    ticker_symbols = get_ticker_symbols(num_ticker_symbols)
    final_df = pd.DataFrame()
    for i, symbol in enumerate(ticker_symbols):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="max").reset_index()

            # Create Labels
            df["Next Day Movement"] = df["Close"] < df[
                "Close"
            ].shift(-1)

            df["Next Day Close"] = df["Close"].shift(-1)

            # trim NaNs
            df = df.dropna()

            # convert types
            df = df.astype({"Next Day Movement": "int"})

            final_df = pd.concat([final_df, df])
        except KeyError:
            print(f"Couldn't find {symbol}. Continuing...")
        print(
            f"Collecting the most recent financial data. Progress: {round((i+1)/len(ticker_symbols) *100, 2)}%.",
            end="\r",
        )
    print("\nDone!")

    test_start = int(len(final_df) * (1-test_split))
    val_start = int(test_start - (len(final_df) * validation_split))

    train_df = final_df.loc[:val_start].copy()
    val_df = final_df.loc[val_start:test_start].copy()
    test_df = final_df.loc[test_start:].copy()

    # print(test_df)
    return train_df, val_df, test_df


def normalize_pre_processed_data(train_df, val_df, test_df):
    # normalize feature columns
    target_columns = ["Next Day Movement", "Next Day Close"]
    feature_columns = train_df.columns.difference([target_columns[0]])

    norm_train_df = train_df.copy()
    norm_val_df = val_df.copy()
    norm_test_df = test_df.copy()
    for c in feature_columns:
        train_mean = train_df[c].mean()
        train_stddev = train_df[c].std()

        norm_train_df[c] = (train_df[c] - train_mean) / train_stddev
        norm_val_df[c] = (val_df[c] - train_mean) / train_stddev
        norm_test_df[c] = (test_df[c] - train_mean) / train_stddev

    return norm_train_df, norm_val_df, norm_test_df, feature_columns, target_columns


def get_ticker_symbols(num_ticker_symbols):
    # Get s&p 500 ticker symbols from wikipedia
    resp = requests.get(
        "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"class": "wikitable sortable"})

    ticker_symbols = []

    for row in table.findAll("tr")[1:num_ticker_symbols + 1]:
        ticker = row.findAll("td")[0].text.strip()
        ticker_symbols.append(ticker)

    ticker_symbols = SINGLE_TICKER_SYMBOL
    return ticker_symbols
