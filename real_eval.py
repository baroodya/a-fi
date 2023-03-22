import numpy as np


def real_movement_eval(model, test_df, test_loader, starting_value=100, sequence_sep=0):
    starting_shares = 0
    # print(test_df)
    curr_value = starting_value
    curr_shares = starting_shares

    i = sequence_sep

    day_data = test_df.iloc[i]
    close = day_data["Close"]
    last_close = close
    hold_curr_shares = curr_value / close

    rise_confidences = []
    fall_confidences = []
    for feat, _ in test_loader:
        day_data = test_df.iloc[i]
        # print(day_data)
        close = day_data["Close"]

        output = model.forward(feat).item()
        confidence = 2 * abs(output - 0.5)
        if output > 0.5:
            rise_confidences.append(confidence)
            curr_shares += confidence * (curr_value / close)
            curr_value -= confidence * (curr_value)
        else:
            fall_confidences.append(confidence)
            curr_value += confidence * (curr_shares * close)
            curr_shares -= confidence * (curr_shares)
        print(f"Yesterday's Close: {last_close:.2f} Today's Pred: {output:.2f} Today's Close: {close:.2f} Shares: {curr_shares:.2f} Cash: {curr_value:.2f} Value {curr_shares * close + curr_value:.2f} Hold Value: {hold_curr_shares * close:.2f}")
        i += 1
        last_close = close
    curr_value += curr_shares * test_df.iloc[-1]["Close"]
    hold_curr_value = hold_curr_shares * test_df.iloc[-1]["Close"]

    print(
        f"{len(rise_confidences)} Rises. Mean: {np.mean(rise_confidences)} Std Dev: {np.std(rise_confidences)}")
    print(
        f"{len(fall_confidences)} Falls. Mean: {np.mean(fall_confidences)} Std Dev: {np.std(fall_confidences)}")
    return hold_curr_value, curr_value


def price_check(model, test_df, test_loader, starting_value=100, sequence_sep=0):
    starting_shares = 0

    curr_value = starting_value
    curr_shares = starting_shares

    i = sequence_sep
    close = test_df.iloc[i]["Close"]
    last_close = close
    last_pred_close = close

    hold_curr_shares = curr_value / close
    rise_count = 0
    fall_count = 0

    for feat, _ in test_loader:
        if (i == len(test_df)):
            break

        last_close = test_df.iloc[i - 1]["Close"]

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
        # print(f"Yesterday's Close: {last_close:.2f} Today's Pred: {output:.2f} Today's Close: {test_df.iloc[i]['Close']:.2f} Shares: {curr_shares:.2f} Cash: {curr_value:.2f} Value {curr_shares * test_df.iloc[i]['Close'] + curr_value:.2f} Hold Value: {hold_curr_shares * test_df.iloc[i]['Close']:.2f}")
        i += 1
        last_pred_close = output
    curr_value += curr_shares * test_df.iloc[-1]["Close"]
    hold_curr_value = hold_curr_shares * test_df.iloc[-1]["Close"]

    print(
        f"Rises: {rise_count}")
    print(
        f"Falls: {fall_count}")
    return hold_curr_value, curr_value
