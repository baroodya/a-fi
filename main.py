import itertools
from datetime import datetime
import json
import matplotlib.pyplot as plt

from constants import (
    MOVEMENT_MODEL_PATH,
    PRICE_MODEL_PATH,
    TRAINING_WEIGHTS_FILE_NAME,
    VAL_WEIGHTS_FILE_NAME,
    TRAIN_STATS_FILE_NAME,
    VAL_STATS_FILE_NAME,
)
from real_eval import real_movement_eval, price_check
from dataset import FeatureDataset
from framework import BaseFramework
import parser
from preprocessing import (pre_process_data, normalize_pre_processed_data)

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
weight_decays = args.weight_decay
epochs_arr = args.epochs
days_prior_arr = args.days_prior
use_pretrained = args.use_pretrained
num_hidden_units_arr = args.num_hidden_units
num_ticker_symbols = args.num_ticker_symbols[0]

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
        weight_decay,
        epochs,
        days_prior,
        architecture,
        num_hidden_units,
    ) in enumerate(get_hyperparameter_combos([
        training_batch_sizes,
        learning_rates,
        weight_decays,
        epochs_arr,
        days_prior_arr,
        architectures,
        num_hidden_units_arr,
    ])):
        print(f"""
-------------------------------------------------------------------------------------
Hyperparameters for Version {i+1}:
Training Batch Size: {training_batch_size}
Learning Rate: {learning_rate}
Number of Epochs: {epochs}
History Considered: {days_prior}
Number of Hidden Units: {num_hidden_units}
Architecture: {architecture.__name__}
-------------------------------------------------------------------------------------

        """)
        # -----------------------------------------------------------------------------------------#
        # Create the datasets                                                                      #
        # -----------------------------------------------------------------------------------------#
        unnormalized_train_df, unnormalized_val_df, unnormalized_test_df = pre_process_data(
            num_ticker_symbols=num_ticker_symbols,
            validation_split=validation_split,
            test_split=test_split,
        )
        training_df, val_df, _, feature_columns, target_columns = normalize_pre_processed_data(
            unnormalized_train_df, unnormalized_val_df, unnormalized_test_df)

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
            num_features=num_features, hidden_units=num_hidden_units)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        model.to(device)

        loss_fn = torch.nn.MSELoss()
        if predict_movement:
            loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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

        val_data = framework.eval(
            val_loader, predict_movement=predict_movement)
        print(
            f"Validation done. Accuracy: {round(val_data['accuracy'] * 100, 3)}%"
        )

        # -----------------------------------------------------------------------------------------#
        # Update best stats and weights                                                            #
        # -----------------------------------------------------------------------------------------#

        current_model_path = PRICE_MODEL_PATH
        if predict_movement:
            current_model_path = MOVEMENT_MODEL_PATH

        # TODO: Make this a utility function for cleanliness
        with open(current_model_path + TRAIN_STATS_FILE_NAME, 'r+') as f:
            best_data = json.load(f)

            if train_data['accuracy'] > best_data["accuracy"]:
                torch.save(
                    model.state_dict(),
                    current_model_path + TRAINING_WEIGHTS_FILE_NAME,
                )
                best_data["accuracy"] = train_data["accuracy"]
                best_data["model_name"] = model.__class__.__name__
                best_data["days_prior"] = days_prior
                best_data["hidden_units"] = num_hidden_units
                best_data["date"] = datetime.now().strftime(
                    "%d/%m/%Y, %H:%M:%S")

            f.seek(0)
            f.truncate()
            json.dump(best_data, f)

        with open(current_model_path + VAL_STATS_FILE_NAME, 'r+') as f:
            best_data = json.load(f)

            if val_data['accuracy'] > best_data["accuracy"]:
                torch.save(
                    model.state_dict(),
                    current_model_path + VAL_WEIGHTS_FILE_NAME,
                )
                best_data["accuracy"] = train_data["accuracy"]
                best_data["model_name"] = model.__class__.__name__
                best_data["days_prior"] = days_prior
                best_data["hidden_units"] = num_hidden_units
                best_data["date"] = datetime.now().strftime(
                    "%d/%m/%Y, %H:%M:%S")

            f.seek(0)
            f.truncate()
            json.dump(best_data, f)

# TODO: Make loading for testing and training cleaner and more space efficient
train_df, val_df, unnormalized_test_df = pre_process_data(
    num_ticker_symbols=num_ticker_symbols,
    validation_split=validation_split,
    test_split=test_split,
)

_, _, test_df, feature_columns, target_columns = normalize_pre_processed_data(
    train_df, val_df, unnormalized_test_df)


current_model_path = PRICE_MODEL_PATH
if predict_movement:
    current_model_path = MOVEMENT_MODEL_PATH
with open(current_model_path + VAL_STATS_FILE_NAME, 'r') as f:
    best_data = json.load(f)
model_class = globals()[best_data["model_name"]]
days_prior = best_data["days_prior"]
num_hidden_units = best_data["hidden_units"]

model = model_class(len(feature_columns), num_hidden_units)
model.load_state_dict(torch.load(current_model_path + VAL_WEIGHTS_FILE_NAME))

target_idx = 1
if predict_movement:
    target_idx = 0

test_dataset = FeatureDataset(
    dataframe=test_df, features=feature_columns, target=target_columns[target_idx], sequence_length=days_prior)
test_loader = DataLoader(
    test_dataset, batch_size=1, shuffle=False)

# TODO: Remove need for optimizer
loss_fn = torch.nn.MSELoss()
if predict_movement:
    loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0)
framework = BaseFramework(
    model=model, loss_function=loss_fn, optimizer=optimizer)

test_data = framework.eval(
    test_loader, predict_movement=predict_movement, threshold=1)
print(
    f"Testing done using {model.__class__.__name__}. Accuracy: {round(test_data['accuracy'] * 100, 3)}%"
)
starting_value = 100
eval = price_check
if predict_movement:
    eval = real_movement_eval
regular_strat, model_based_strat = eval(
    model, unnormalized_test_df, test_loader, starting_value)
print(f"If you invested ${starting_value}, you would end with ${model_based_strat:.2f}. This is {(model_based_strat - regular_strat) / regular_strat * 100:.2f}% more than just buying and holding.")
