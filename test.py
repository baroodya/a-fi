from constants import (
    MOVEMENT_MODEL_PATH,
    PRICE_MODEL_PATH,
    VAL_WEIGHTS_FILE_NAME,
    VAL_STATS_FILE_NAME,
)
from dataset import FeatureDataset
from framework import BaseFramework
import json
import torch
from torch.utils.data import DataLoader


def eval_on_test_data(feature_columns, target_columns, norm_test_df, predict_movement):
    # Load the best model so far
    current_model_path = PRICE_MODEL_PATH
    if predict_movement:
        current_model_path = MOVEMENT_MODEL_PATH
    with open(current_model_path + VAL_STATS_FILE_NAME, 'r') as f:
        best_data = json.load(f)
    model_class = globals()[best_data["model_name"]]
    days_prior = best_data["days_prior"]
    num_hidden_units = best_data["hidden_units"]
    sequence_sep = best_data["sequence_sep"]

    model = model_class(len(feature_columns), num_hidden_units)
    model.load_state_dict(torch.load(
        current_model_path + VAL_WEIGHTS_FILE_NAME))

    target_idx = 1
    if predict_movement:
        target_idx = 0

    test_dataset = FeatureDataset(
        dataframe=norm_test_df, features=feature_columns, target=target_columns[target_idx], sequence_length=days_prior, sequence_sep=sequence_sep)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False)

    loss_fn = torch.nn.MSELoss()
    if predict_movement:
        loss_fn = torch.nn.BCELoss()
    framework = BaseFramework(
        model=model, loss_function=loss_fn)

    test_data = framework.eval(
        test_loader, predict_movement=predict_movement)
    print(
        f"Testing done using {model.__class__.__name__}. Accuracy: {test_data['accuracy'] * 100:.2f}%"
    )
