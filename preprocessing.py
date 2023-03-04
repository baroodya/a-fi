import pandas as pd
import yfinance as yf


def pre_process_data(ticker_symbols, validation_split, test_split):
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
    print(final_df)

    test_start = int(len(final_df) * (1-test_split))
    val_start = int(test_start - (len(final_df) * validation_split))

    train_df = final_df.loc[:val_start].copy()
    val_df = final_df.loc[val_start:test_start].copy()
    test_df = final_df.loc[test_start:].copy()

    # normalize feature columns
    target_columns = ["Next Day Movement", "Next Day Close"]
    feature_columns = train_df.columns.difference(target_columns)
    for c in feature_columns:
        train_mean = train_df[c].mean()
        train_stddev = train_df[c].std()

        train_df[c] = (train_df[c] - train_mean) / train_stddev
        val_df[c] = (val_df[c] - train_mean) / train_stddev
        test_df[c] = (test_df[c] - train_mean) / train_stddev

    return train_df, val_df, test_df, feature_columns, target_columns