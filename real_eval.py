import numpy as np
import torch

def real_eval(model, df, norm_df, feature_columns, target_column, starting_value=100, sequence_length=5, norm_hist_length=5, sequence_sep=0):
    starting_shares = 0

    df = df.iloc[(len(df) - len(norm_df)):]

    curr_value = starting_value
    curr_shares = starting_shares

    next_day_closes = df[target_column]
    today_close = next_day_closes[sequence_length-1]

    last_pred_close = 0

    hold_curr_shares = curr_value / today_close
    rise_count = 0
    fall_count = 0

    for i, next_close in enumerate(next_day_closes[:len(df) - sequence_length - min(0,sequence_sep)]):
        if i % (sequence_sep + 1) == 0:
            # Collect and normalize recent data
            norm_data = df.iloc[i-(norm_hist_length-sequence_length):i+sequence_length]
            recent_data = df.iloc[i:i+sequence_length]
            norm_recent_data = (recent_data - norm_data.mean())/norm_data.std()
            feat = torch.tensor(norm_recent_data[feature_columns].values, dtype=torch.float32)
            feat = torch.unsqueeze(feat, 0)

            output = model.forward(feat).item()
            confidence = 1
            if output > last_pred_close:
                rise_count += 1
                curr_shares += confidence * (curr_value / today_close)
                curr_value -= confidence * (curr_value)
            else:
                fall_count += 1
                curr_value += confidence * (curr_shares * today_close)
                curr_shares -= confidence * (curr_shares)
            # print(f"Yesterday's Close: {last_close:.2f} Today's Pred: {output:.2f} Today's Close: {closes[i]:.2f} Shares: {curr_shares:.2f} Cash: {curr_value:.2f} Value {curr_shares * closes[i] + curr_value:.2f} Hold Value: {hold_curr_shares * closes[i]:.2f}")
            last_pred_close = output
        i += 1
        today_close = next_close
    curr_value += curr_shares * today_close
    hold_curr_value = hold_curr_shares * today_close

    print(
        f"Rises: {rise_count}")
    print(
        f"Falls: {fall_count}")
    return hold_curr_value, curr_value
