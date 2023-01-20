import torch.nn as nn


class AFiMovementModel(nn.Module):
    def __init__(self, input_size, loss_function):
        super(AFiMovementModel, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
            nn.Sigmoid(),
            nn.Flatten(0, 1),
        )

        self.loss_func = loss_function

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    def train(self, train_loader, optimizer, epochs):
        running_loss = 0
        for epoch in range(epochs):
            # batch training
            batch_num = 1
            for features, target in train_loader:
                self.zero_grad()

                # forward pass
                output = self(features)
                loss = self.loss_func(output, target)

                # backward pass
                loss.backward()
                optimizer.step()

                # Print updates every 100 batches
                if batch_num % 100 == 0:
                    progress = round(
                        (batch_num + (epoch * len(train_loader)))
                        / (len(train_loader) * epochs)
                        * 100,
                        2,
                    )
                    print(
                        f"Training on {len(train_loader)} datapoints. Progress: {progress}%. Avg. Loss: {running_loss / (batch_num + (epoch * len(train_loader)))}",
                        end="\r",
                    )
                batch_num += 1
                running_loss += loss.item()

        print()

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
