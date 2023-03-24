from constants import (
    MODEL_PATH,
    TRAINING_WEIGHTS_FILE_NAME,
    VAL_WEIGHTS_FILE_NAME,
    TRAIN_STATS_FILE_NAME,
    VAL_STATS_FILE_NAME,
    TRAINING_MODEL_FILE_NAME,
    VAL_MODEL_FILE_NAME,
)
from datetime import datetime
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import time


class BaseFramework():
    def __init__(self, model, loss_function, ticker_symbol):
        self.model = model
        self.loss_func = loss_function
        self.ticker_symbol = ticker_symbol

        self.train_data = {}
        self.val_data = {}

    def train(self, train_loader, epochs, optimizer):
        self.model.train()
        losses = []
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.9)

        # plt.figure()
        # plt.xlabel("Number of Examples")
        # plt.ylabel("Log Loss")
        # plt.title(f"Loss for {self.ticker_symbol}")
        # plt.show()

        recent_idx = 0
        for i in range(epochs):
            # batch training
            running_loss = 0
            self.model.train()
            for features, target in train_loader:
                # forward pass
                prediction = self.model.forward(features)
                loss = self.loss_func(prediction, target)
                # print(
                    # f"Prediction: {prediction}\nTarget: {target}\nLoss: {loss}\n")
                # time.sleep(10)
                losses.append(loss.item())
                running_loss += loss.item()

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            print(
                        f"Training on {self.ticker_symbol}. Progress: {((i+1) / epochs) * 100:.2f}%. Avg. Loss: {running_loss / len(train_loader) :.5f}.",
                        end="\r",
                    )
        print()
        return losses

    def eval(self, loader, threshold=1, is_training_data=False):
        num_correct = 0
        num_seen = 0
        running_loss = 0
        self.model.eval()

        outputs = []
        for features, target in loader:
            batch_output = self.model.forward(features)
            running_loss += self.loss_func(batch_output, target)

            for output, target in zip(batch_output, target):
                outputs.append(output.item())
                if abs(output - target) < threshold:
                    num_correct += 1
                num_seen += 1
        store = {
            "accuracy": num_correct / num_seen,
            "loss": running_loss / len(loader),
            "predictions": outputs,
        }
        if is_training_data:
            self.train_data = store
        else:
            self.val_data = store
        return store

    def save_model(self, days_prior, num_hidden_units, sequence_sep, is_training=True):
        self.model.eval()
        cwd = os.getcwd()
        current_model_path =  os.path.join(cwd, MODEL_PATH)
        current_model_path = os.path.join(current_model_path, self.ticker_symbol)
        if not os.path.exists(current_model_path):
            os.mkdir(current_model_path)

        stats_file_name = VAL_STATS_FILE_NAME
        weights_file_name = VAL_WEIGHTS_FILE_NAME
        model_file_name = VAL_MODEL_FILE_NAME
        if is_training:
            stats_file_name = TRAIN_STATS_FILE_NAME
            weights_file_name = TRAINING_WEIGHTS_FILE_NAME
            model_file_name = TRAINING_MODEL_FILE_NAME


        file_path = os.path.join(current_model_path, stats_file_name)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(current_model_path, weights_file_name),
                )
                torch.save(self.model, os.path.join(current_model_path, model_file_name))

                best_data = {}
                if is_training:
                    best_data["accuracy"] = self.train_data["accuracy"]
                else:
                    best_data["accuracy"] = self.val_data["accuracy"]
                best_data["model_name"] = self.model.__class__.__name__
                best_data["days_prior"] = days_prior
                best_data["hidden_units"] = num_hidden_units
                best_data["sequence_sep"] = sequence_sep
                best_data["date"] = datetime.now().strftime(
                    "%d/%m/%Y, %H:%M:%S")
                
                f.seek(0)
                f.truncate()
                json.dump(best_data, f)
        else:
            with open(file_path, 'r+') as f:
                best_data = json.load(f)

                if is_training and (self.train_data['accuracy'] > best_data["accuracy"]) or not is_training and (self.val_data['accuracy'] > best_data["accuracy"]):
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(current_model_path, weights_file_name),
                    )
                    torch.save(self.model, os.path.join(current_model_path, model_file_name))

                    
                    if is_training:
                        best_data["accuracy"] = self.train_data["accuracy"]
                    else:
                        best_data["accuracy"] = self.val_data["accuracy"]
                    best_data["model_name"] = self.model.__class__.__name__
                    best_data["days_prior"] = days_prior
                    best_data["hidden_units"] = num_hidden_units
                    best_data["sequence_sep"] = sequence_sep
                    best_data["date"] = datetime.now().strftime(
                        "%d/%m/%Y, %H:%M:%S")

                f.seek(0)
                f.truncate()
                json.dump(best_data, f)