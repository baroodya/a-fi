import numpy as np
import torch

def real_eval(model, df, feature_columns, target_column, starting_value=100, sequence_length=5, sequence_sep=0):
    starting_shares = 0

    curr_value = starting_value
    curr_shares = starting_shares

    next_day_closes = df[target_column]
    today_close = next_day_closes[sequence_length-1]

    last_pred_close = 0

    hold_curr_shares = curr_value / today_close
    rise_count = 0
    fall_count = 0

    i = sequence_length
    for next_close in next_day_closes[sequence_length+sequence_sep:]:
        if i % sequence_sep == 0:
            # Collect and normalize recent data
            recent_data = df.iloc[i-sequence_length:i]
            norm_recent_data = (recent_data - recent_data.mean())/recent_data.std()
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
