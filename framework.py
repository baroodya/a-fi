import time


class BaseFramework():
    def __init__(self, model, loss_function):
        self.model = model

        self.loss_func = loss_function

    def train(self, train_loader, epochs, optimizer):
        losses = []
        total_batch_count = 0
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

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print updates every 100 batches
                if batch_num % 10 == 0:
                    progress = (total_batch_count) / \
                        (len(train_loader) * epochs) * 100
                    print(
                        f"Training on {epochs * len(train_loader)} batches. Progress: {progress:.2f}%. Avg. Loss: {running_loss / batch_num :.5f}.",
                        end="\r",
                    )
                    batch_num = 1
                    running_loss = 0
                else:
                    batch_num += 1
                    running_loss += loss.item()
                total_batch_count += 1

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
        return {
            "accuracy": num_correct / num_seen,
            "loss": running_loss / len(loader),
            "predictions": outputs,
        }
