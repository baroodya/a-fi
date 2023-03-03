import torch.nn as nn


class AFiMovementModel(nn.Module):
    def __init__(self, input_size, loss_function, model):
        super(AFiMovementModel, self).__init__()
        self.model = model(input_size)
        print(self.model)

        self.loss_func = loss_function

    def forward(self, x):
        logits = self.model(x)
        return logits

    def train(self, train_loader, optimizer, epochs):
        losses = []
        for epoch in range(epochs):
            # batch training
            running_loss = 0
            batch_num = 1
            for features, target in train_loader:
                # forward pass
                prediction = self.forward(features)
                loss = self.loss_func(prediction, target)
                # print(
                #     f"Prediction: {prediction}\nTarget: {target}\nLoss: {loss}\n")

                losses.append(loss.item())

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print updates every 100 batches
                if batch_num % 100 == 0:
                    progress = round(
                        (batch_num + 1 + (epoch * len(train_loader)))
                        / (len(train_loader) * epochs)
                        * 100,
                        2,
                    )
                    print(
                        f"Training on {epochs * len(train_loader)} batches. Progress: {progress}%. Avg. Loss: {round(running_loss / batch_num, 5)}. Last output: {prediction[0].item()}.",
                        end="\r",
                    )
                    batch_num = 1
                    running_loss = 0
                else:
                    batch_num += 1
                    running_loss += loss.item()

        print()
        return losses

    def test(self, loader):
        num_correct = 0
        num_seen = 0
        running_loss = 0
        for features, label in loader:
            output = self(features)
            running_loss += self.loss_func(output, label)

            if (
                output > 0.5
                and label == 1
                or output < 0.5
                and label == 0
            ):
                num_correct += 1
            num_seen += 1
        return {
            "accuracy": num_correct / num_seen,
            "loss": running_loss / len(loader),
        }
