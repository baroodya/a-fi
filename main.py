import itertools
import matplotlib.pyplot as plt
import re

from constants import (
    MOVEMENT_MODEL_PATH,
    PRICE_MODEL_PATH,
    TRAINING_WEIGHTS_FILE_NAME,
    VAL_WEIGHTS_FILE_NAME,
    TEST_WEIGHTS_FILE_NAME,
    STATS_FILE_NAME,
    SINGLE_TICKER_SYMBOL,
)
from dataset import FeatureDataset
from framework import BaseFramework
import parser
from preprocessing import pre_process_data

from movement_prediction.architectures import (
    MovementShallowRegressionLSTM
)
from price_prediction.architectures import (
    ShallowRegressionLSTM
)

import torch
from torch.utils.data import DataLoader
# -----------------------------------------------------------------------------------------#
# Collect Hyperparameters                                                                  #
# -----------------------------------------------------------------------------------------#
args = parser.parse_args()

predict_movement = args.predict_movement
training_batch_sizes = args.batch_size
validation_split = args.val_split[0]
test_split = args.test_split[0]
shuffle_dataset = args.shuffle_dataset
learning_rates = args.learning_rate
epochs_arr = args.epochs
days_prior_arr = args.days_prior
use_pretrained = args.use_pretrained
hidden_units = 16
num_ticker_symbols = args.num_ticker_symbols

architectures = []
if predict_movement:
    architectures.append(MovementShallowRegressionLSTM)
else:
    architectures.append(ShallowRegressionLSTM)


def get_hyperparameter_combos(*hyperparameters):
    return list(itertools.product(*hyperparameters[0]))


if not use_pretrained:
    for i, (
        training_batch_size,
        learning_rate,
        epochs,
        days_prior,
        num_ticker_symbols,
        architecture,
    ) in enumerate(get_hyperparameter_combos([
        training_batch_sizes,
        learning_rates,
        epochs_arr,
        days_prior_arr,
        num_ticker_symbols,
        architectures,
    ])):
        print(f"""
-------------------------------------------------------------------------------------
Hyperparameters for Version {i+1}:
Training Batch Size: {training_batch_size}
Learning Rate: {learning_rate}
Number of Epochs: {epochs}
History Considered: {days_prior}
Number of Ticker Symbols: {num_ticker_symbols}
Architecture: {architecture.__name__}
-------------------------------------------------------------------------------------

        """)
        # -----------------------------------------------------------------------------------------#
        # Create the datasets                                                                      #
        # -----------------------------------------------------------------------------------------#
        training_df, val_df, _, feature_columns, target_columns = pre_process_data(
            num_ticker_symbols=num_ticker_symbols,
            validation_split=validation_split,
            test_split=test_split,
        )

        num_features = len(feature_columns)
        target_idx = 1
        if predict_movement:
            target_idx = 0

        training_dataset = FeatureDataset(
            dataframe=training_df, features=feature_columns, target=target_columns[target_idx], sequence_length=days_prior)

        val_dataset = FeatureDataset(
            dataframe=val_df, features=feature_columns, target=target_columns[target_idx], sequence_length=days_prior)

        # for repeatability
        torch.manual_seed(99)

        training_loader = DataLoader(
            training_dataset, batch_size=training_batch_size, shuffle=shuffle_dataset)
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=shuffle_dataset)

        # -----------------------------------------------------------------------------------------#
        # Model, Optimizer, Loss                                                                   #
        # -----------------------------------------------------------------------------------------#
        model = architectures[0](
            num_features=num_features, hidden_units=hidden_units)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        model.to(device)

        loss_fn = torch.nn.MSELoss()
        if predict_movement:
            loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # -----------------------------------------------------------------------------------------#
        # Train the model                                                                          #
        # -----------------------------------------------------------------------------------------#
        framework = BaseFramework(
            model=model, loss_function=loss_fn, optimizer=optimizer)

        losses = framework.train(train_loader=training_loader, epochs=epochs)

        # -----------------------------------------------------------------------------------------#
        # Evaluate the model                                                                       #
        # -----------------------------------------------------------------------------------------#
        train_data = framework.eval(
            training_loader, predict_movement=predict_movement)
        print(
            f"Training done. Accuracy: {round(train_data['accuracy'] * 100, 3)}%"
        )

        validation_data = framework.eval(
            val_loader, predict_movement=predict_movement)
        print(
            f"Validation done. Accuracy: {round(validation_data['accuracy'] * 100, 3)}%"
        )

        # -----------------------------------------------------------------------------------------#
        # Update best stats and weights                                                            #
        # -----------------------------------------------------------------------------------------#

        current_model_path = PRICE_MODEL_PATH
        if predict_movement:
            current_model_path = MOVEMENT_MODEL_PATH

        best_training_acc = 0.0
        best_val_acc = 0.0

        with open(current_model_path + STATS_FILE_NAME, 'r+') as f:
            lines = f.readlines()

            best_training_acc = float(re.findall(r"\d+\.\d+", lines[0])[0])
            best_training_model = re.findall(
                r"(?<=\btraining:\s)\w+", lines[0])[0]
            best_val_acc = float(re.findall(r"\d+\.\d+", lines[1])[0])
            best_val_model = re.findall(
                r"(?<=\bvalidation:\s)(\w+)", lines[1])[0]

            if train_data['accuracy'] > best_training_acc:
                torch.save(
                    model.state_dict(),
                    current_model_path + TRAINING_WEIGHTS_FILE_NAME,
                )
                best_training_acc = train_data["accuracy"]
                best_training_model = model.__class__.__name__

            if validation_data['accuracy'] > best_val_acc:
                torch.save(
                    model.state_dict(),
                    current_model_path + VAL_WEIGHTS_FILE_NAME,
                )
                best_val_acc = validation_data["accuracy"]
                best_val_model = model.__class__.__name__

            f.seek(0)
            f.truncate()
            f.write(
                f"training: {best_training_model} {best_training_acc * 100:.2f}%\nvalidation: {best_val_model} {best_val_acc * 100:.2f}%")

_, _, test_df, feature_columns, target_columns = pre_process_data(
    num_ticker_symbols=num_ticker_symbols,
    validation_split=validation_split,
    test_split=test_split,
)

current_model_path = PRICE_MODEL_PATH
if predict_movement:
    current_model_path = MOVEMENT_MODEL_PATH
with open(current_model_path + STATS_FILE_NAME, 'r') as f:
    lines = f.readlines()

    best_val_model = re.findall(
        r"(?<=\bvalidation:\s)(\w+)", lines[1])[0]
model_class = globals()[best_val_model]
model = model_class(len(feature_columns), hidden_units)
model.load_state_dict(torch.load(current_model_path + VAL_WEIGHTS_FILE_NAME))

test_dataset = FeatureDataset(
    dataframe=test_df, features=feature_columns, target=target_columns[target_idx], sequence_length=days_prior)
test_loader = DataLoader(
    test_dataset, batch_size=1, shuffle=shuffle_dataset)
test_data = framework.eval(
    test_loader, predict_movement=predict_movement, threshold=1)
print(
    f"Testing done using {model.__name__}. Accuracy: {round(test_data['accuracy'] * 100, 3)}%"
)
