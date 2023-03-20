from constants import (
    MOVEMENT_MODEL_PATH,
    PRICE_MODEL_PATH,
    TRAINING_WEIGHTS_FILE_NAME,
    VAL_WEIGHTS_FILE_NAME,
    TRAIN_STATS_FILE_NAME,
    VAL_STATS_FILE_NAME,
)
from datetime import datetime
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import torch


class BaseFramework():
    def __init__(self, model, loss_function, ticker_symbol):
        self.model = model
        self.loss_func = loss_function
        self.ticker_symbol = ticker_symbol

    def train(self, train_loader, epochs, optimizer):
        losses = []
        avg_losses = []
        recent_losses = np.zeros(100)
        total_batch_count = 0
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.9)

        plt.figure()
        plt.xlabel("Number of Examples")
        plt.ylabel("Log Loss")
        plt.title(f"Loss for {self.ticker_symbol}")
        plt.show()

        recent_idx = 0
        for _ in range(epochs):
            # batch training
            running_loss = 0
            batch_num = 1
            self.model.train()
            for features, target in train_loader:
                # forward pass
                prediction = self.model.forward(features)
                loss = self.loss_func(prediction, target)
                # print(
                #     f"Prediction: {prediction}\nTarget: {target}\nLoss: {loss}\n")
                # time.sleep(10)
                losses.append(loss.item())
                recent_losses[recent_idx] = loss.item()

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print updates every 100 batches
                if batch_num % 100 == 0:
                    # avg_losses.append(np.average(recent_losses))
                    # recent_losses = np.zeros(100)
                    # plt.plot(np.arange(len(avg_losses)), np.log(avg_losses), 'C0')
                    # plt.pause(0.001)

                    progress = (total_batch_count) / \
                        (len(train_loader) * epochs) * 100
                    print(
                        f"Training on {self.ticker_symbol}. Progress: {progress:.2f}%. Avg. Loss: {running_loss / batch_num :.5f}.",
                        end="\r",
                    )
                    batch_num = 1
                    running_loss = 0
                else:
                    batch_num += 1
                    running_loss += loss.item()
                total_batch_count += 1
            scheduler.step()
        print()
        return losses

    def eval(self, loader, predict_movement=False, threshold=0.1):
        num_correct = 0
        num_seen = 0
        running_loss = 0
        self.model.eval()

        outputs = []
        for features, label in loader:
            batch_output = self.model.forward(features)
            running_loss += self.loss_func(batch_output, label)

            for output, label in zip(batch_output, label):
                # print(output.item(), label.item())
                outputs.append(output.item())
                if predict_movement:
                    if (
                        output > 0.5
                        and label == 1
                        or output < 0.5
                        and label == 0
                    ):
                        num_correct += 1

                else:
                    if abs(output - label) < threshold:
                        num_correct += 1
                num_seen += 1
        self.train_data = {
            "accuracy": num_correct / num_seen,
            "loss": running_loss / len(loader),
            "predictions": outputs,
        }
        return self.train_data

    def save_model(self, days_prior, num_hidden_units, sequence_sep, predict_movement=False, is_training=True):
        cwd = os.getcwd()
        current_model_path =  os.path.join(cwd, PRICE_MODEL_PATH)
        if predict_movement:
            current_model_path = os.path.join(cwd, MOVEMENT_MODEL_PATH)
        current_model_path = os.path.join(current_model_path, self.ticker_symbol)
        if not os.path.exists(current_model_path):
            os.mkdir(current_model_path)

        stats_file_name = VAL_STATS_FILE_NAME
        weights_file_name = VAL_WEIGHTS_FILE_NAME
        if is_training:
            stats_file_name = TRAIN_STATS_FILE_NAME
            weights_file_name = TRAINING_WEIGHTS_FILE_NAME

        file_path = os.path.join(current_model_path, stats_file_name)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(current_model_path, weights_file_name),
                )
                best_data = {}
                best_data["accuracy"] = self.train_data["accuracy"]
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

                if self.train_data['accuracy'] > best_data["accuracy"]:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(current_model_path, weights_file_name),
                    )
                    best_data["accuracy"] = self.train_data["accuracy"]
                    best_data["model_name"] = self.model.__class__.__name__
                    best_data["days_prior"] = days_prior
                    best_data["hidden_units"] = num_hidden_units
                    best_data["sequence_sep"] = sequence_sep
                    best_data["date"] = datetime.now().strftime(
                        "%d/%m/%Y, %H:%M:%S")

                f.seek(0)
                f.truncate()
                json.dump(best_data, f)