import numpy as np


def real_movement_eval(model, test_df, test_loader, starting_value=100):
    starting_shares = 0
    # print(test_df)
    curr_value = starting_value
    curr_shares = starting_shares
    i = 0
    day_data = test_df.iloc[i]
    close = day_data["Close"]
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
        i += 1
    curr_value += curr_shares * test_df.iloc[-1]["Close"]
    hold_curr_value = hold_curr_shares * test_df.iloc[-1]["Close"]

    # print(
    #     f"Rises. Mean: {np.mean(rise_confidences)} Std Dev: {np.std(rise_confidences)}")
    # print(
    #     f"Falls. Mean: {np.mean(fall_confidences)} Std Dev: {np.std(fall_confidences)}")
    return hold_curr_value, curr_value


def price_check(model, test_df, test_loader, starting_value=100, sequence_sep=0):
    starting_shares = 0
    # print(test_df)
    curr_value = starting_value
    curr_shares = starting_shares
    i = 1 + sequence_sep
    day_data = test_df.iloc[i]
    close = day_data["Close"]
    hold_curr_shares = curr_value / close
    rise_confidences = []
    fall_confidences = []
    for feat, _ in test_loader:
        normalized_last_close = test_df.iloc[i -
                                             1-sequence_sep]["Normalized Close"]
        if (i == len(test_df)):
            continue
        day_data = test_df.iloc[i-1-sequence_sep]
        # print(day_data)
        close = day_data["Close"]

        output = model.forward(feat).item()
        confidence = 1
        if output > normalized_last_close:
            rise_confidences.append(confidence)
            curr_shares += confidence * (curr_value / close)
            curr_value -= confidence * (curr_value)
        else:
            fall_confidences.append(confidence)
            curr_value += confidence * (curr_shares * close)
            curr_shares -= confidence * (curr_shares)
        i += 1
    curr_value += curr_shares * test_df.iloc[-1]["Close"]
    hold_curr_value = hold_curr_shares * test_df.iloc[-1]["Close"]

    # print(
    #     f"Rises. Mean: {np.mean(rise_confidences)} Std Dev: {np.std(rise_confidences)}")
    # print(
    #     f"Falls. Mean: {np.mean(fall_confidences)} Std Dev: {np.std(fall_confidences)}")
    return hold_curr_value, curr_value
