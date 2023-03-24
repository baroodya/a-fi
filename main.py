import itertools
import matplotlib.pyplot as plt
import pandas as pd

from real_eval import real_eval
from dataset import FeatureDataset
from framework import BaseFramework
import parser
from preprocessing import DataPreprocessor, get_ticker_symbols

from architectures import (
    ShallowRegressionLSTM,
    DoubleRegressionLSTM,
    QuadRegressionLSTM,
    DeepRegressionLSTM,
)

import torch
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------------------#
# Collect Hyperparameters                                                                  #
# -----------------------------------------------------------------------------------------#
args = parser.parse_args()

training_batch_sizes = args.batch_size
validation_split = args.val_split[0]
test_split = args.test_split[0]
shuffle_dataset = args.shuffle_dataset
learning_rates = args.learning_rate
weight_decays = args.weight_decay
epochs_arr = args.epochs
days_prior_arr = args.days_prior
sequence_seps = args.sequence_sep
use_pretrained = args.use_pretrained
num_hidden_units_arr = args.num_hidden_units
num_ticker_symbols = args.num_ticker_symbols[0]
test_best = args.test_best

architectures = []
architectures.append(ShallowRegressionLSTM)
# architectures.append(DoubleRegressionLSTM)
# architectures.append(QuadRegressionLSTM)
# architectures.append(DeepRegressionLSTM)


def get_hyperparameter_combos(*hyperparameters):
    return list(itertools.product(*hyperparameters[0]))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

starting_value = 100


ticker_symbols = get_ticker_symbols(1)
train_dfs = {}
norm_train_dfs = {}
val_dfs = {}
norm_val_dfs = {}
test_dfs = {}
norm_test_dfs = {}

# -----------------------------------------------------------------------------------------#
# Preprocess the Data                                                                      #
# -----------------------------------------------------------------------------------------#
for ticker_symbol in ticker_symbols:
    preprocessor = DataPreprocessor(
        ticker_symbol=ticker_symbol,
        validation_split=validation_split,
        test_split=test_split
    )
    preprocessor.pre_process_data()
    train_dfs[ticker_symbol], val_dfs[ticker_symbol], test_dfs[ticker_symbol] = preprocessor.get_dfs()
    preprocessor.normalize_pre_processed_data()
    norm_train_dfs[ticker_symbol], norm_val_dfs[ticker_symbol], norm_test_dfs[ticker_symbol] = preprocessor.get_norm_dfs()
    if ticker_symbol == ticker_symbols[0]:
        feature_columns = preprocessor.get_feature_columns()
        target_column = preprocessor.get_target_column()

    num_features = len(feature_columns)

# plt.ion()
if not use_pretrained:
    for i, (
        training_batch_size,
        learning_rate,
        weight_decay,
        epochs,
        days_prior,
        sequence_sep,
        architecture,
        num_hidden_units,
    ) in enumerate(get_hyperparameter_combos([
        training_batch_sizes,
        learning_rates,
        weight_decays,
        epochs_arr,
        days_prior_arr,
        sequence_seps,
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
Sequence Seperation: {sequence_sep}
Number of Hidden Units: {num_hidden_units}
Architecture: {architecture.__name__}
-------------------------------------------------------------------------------------

        """)
        train_acc_sum = 0.0
        val_acc_sum = 0.0
        hold_value_sum = 0.0
        model_value_sum = 0.0
        for ticker_symbol in ticker_symbols:
            train_df = train_dfs[ticker_symbol]
            norm_train_df = norm_train_dfs[ticker_symbol]
            val_df = val_dfs[ticker_symbol]
            norm_val_df = norm_val_dfs[ticker_symbol]
            test_df = test_dfs[ticker_symbol]
            norm_test_df = norm_val_dfs[ticker_symbol]

            # -----------------------------------------------------------------------------------------#
            # Create the datasets and dataloaders                                             #
            # -----------------------------------------------------------------------------------------#
            training_dataset = FeatureDataset(
                dataframe=norm_train_df, features=feature_columns, target=target_column, sequence_length=days_prior, sequence_sep=sequence_sep)

            val_dataset = FeatureDataset(
                dataframe=norm_val_df, features=feature_columns, target=target_column, sequence_length=days_prior, sequence_sep=sequence_sep)

            # for repeatability
            torch.manual_seed(99)

            training_loader = DataLoader(
                training_dataset, batch_size=training_batch_size, shuffle=shuffle_dataset)
            val_loader = DataLoader(
                val_dataset, batch_size=1, shuffle=False)

            # -----------------------------------------------------------------------------------------#
            # Model, Optimizer, Loss                                                                   #
            # -----------------------------------------------------------------------------------------#
            model = architecture(
                sequence_length=days_prior, num_features=num_features, hidden_units=num_hidden_units)
            model.to(device)

            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            # -----------------------------------------------------------------------------------------#
            # Train the model                                                                          #
            # -----------------------------------------------------------------------------------------#
            framework = BaseFramework(
                model=model, loss_function=loss_fn, ticker_symbol=ticker_symbol)

            losses = framework.train(
                train_loader=training_loader, epochs=epochs, optimizer=optimizer)

            # -----------------------------------------------------------------------------------------#
            # Evaluate the model                                                                       #
            # -----------------------------------------------------------------------------------------#
            train_data = framework.eval(
                training_loader, is_training_data=True)
            train_acc_sum += train_data['accuracy']
            # print(
            #     f"Training for {ticker_symbol} done. Accuracy: {train_data['accuracy'] * 100:.2f}%"
            # )

            val_data = framework.eval(
                val_loader)
            val_acc_sum += val_data["accuracy"]
            # print(
            #     f"Validation for {ticker_symbol} done. Accuracy: {val_data['accuracy'] * 100:.2f}%"
            # )

            closes = list(val_df["Close"])

            regular_strat, model_based_strat = real_eval(
                model, closes, val_loader, starting_value, sequence_sep)
            hold_value_sum += regular_strat
            model_value_sum += model_based_strat
            # print(f"If you invested ${starting_value} in {ticker_symbol}, you would end with ${model_based_strat:.2f}. This is {(model_based_strat - regular_strat) / regular_strat * 100:.2f}% more than the ${regular_strat:.2f} you would earn by just buying and holding.")

            # -----------------------------------------------------------------------------------------#
            # Plot results                                                                       #
            # -----------------------------------------------------------------------------------------#
            # plt.figure()
            plt.plot(norm_train_df.index.values,
                     norm_train_df["Next Day Close"], label="Ground Truth")
            plt.plot(norm_train_df.index.values,
                     train_data["predictions"], label="Prediction")
            plt.xlabel("Date")
            plt.ylabel("Predicted Values")
            plt.suptitle(f"Training Data Predictions for {ticker_symbol}")
            plt.title(f"Version {i+1}. Architecture: {architecture.__name__}")
            plt.legend()
            plt.show()
            # plt.pause(0.1)

            # plt.figure()
            plt.plot(norm_val_df.index.values,
                     norm_val_df["Next Day Close"], label="Ground Truth")
            plt.plot(norm_val_df.index.values,
                     val_data["predictions"], label="Prediction")
            plt.xlabel("Date")
            plt.ylabel("Predicted Values")
            plt.title(f"Validation Data Predictions for {ticker_symbol}")
            plt.legend()
            plt.show()
            # plt.pause(0.1)

            # -----------------------------------------------------------------------------------------#
            # Update best stats and weights                                                            #
            # -----------------------------------------------------------------------------------------#

            framework.save_model(days_prior, num_hidden_units, sequence_sep, is_training=True)
            framework.save_model(days_prior, num_hidden_units, sequence_sep, is_training=False)

        train_acc = train_acc_sum / len(ticker_symbols) * 100
        val_acc = val_acc_sum / len(ticker_symbols) * 100
        start_date = val_df.index[0].strftime("%B %-d, %Y")
        end_date = val_df.index[-1].strftime("%B %-d, %Y")
        hold_value = hold_value_sum / len(ticker_symbols)
        model_value = model_value_sum / len(ticker_symbols)
        improvement = (model_value - hold_value) / hold_value
        print(f"""
Training done.
Average training accuracy: {train_acc:.2f}%.
Average Validation Accuracy: {val_acc:.2f}%.
If you started with ${starting_value}:
    Buying equal weights on {start_date} and holding would result in ${hold_value:.2f} on {end_date}.
    Buying and Selling equal weights based on the model starting on {start_date} would result in ${model_value:.2f} on {end_date}.
    This is is an average improvement of {improvement * 100:.2f}%.""")
