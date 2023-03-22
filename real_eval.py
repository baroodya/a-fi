import numpy as np

def real_eval(model, closes, test_loader, starting_value=100, sequence_sep=0):
    starting_shares = 0

    curr_value = starting_value
    curr_shares = starting_shares

    i = sequence_sep
    close = closes[i]
    last_close = close
    last_pred_close = close

    hold_curr_shares = curr_value / close
    rise_count = 0
    fall_count = 0

    for feat, _ in test_loader:
        if (i == len(closes)):
            break

        last_close = closes[i - 1]

        output = model.forward(feat).item()
        confidence = 1
        if output > last_pred_close:
            rise_count += 1
            curr_shares += confidence * (curr_value / last_close)
            curr_value -= confidence * (curr_value)
        else:
            fall_count += 1
            curr_value += confidence * (curr_shares * last_close)
            curr_shares -= confidence * (curr_shares)
        print(f"Yesterday's Close: {last_close:.2f} Today's Pred: {output:.2f} Today's Close: {closes[i]:.2f} Shares: {curr_shares:.2f} Cash: {curr_value:.2f} Value {curr_shares * closes[i] + curr_value:.2f} Hold Value: {hold_curr_shares * closes[i]:.2f}")
        i += 1
        last_pred_close = output
    curr_value += curr_shares * closes[-1]
    hold_curr_value = hold_curr_shares * closes[-1]

    print(
        f"Rises: {rise_count}")
    print(
        f"Falls: {fall_count}")
    return hold_curr_value, curr_value
