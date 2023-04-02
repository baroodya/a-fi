import torch


class ShallowRegressionLSTM(torch.nn.Module):
    def __init__(self, sequence_length, num_features, hidden_units):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_features = num_features  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = torch.nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )
        self.relu = torch.nn.ReLU()

        self.linear = torch.nn.Linear(
            in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))

        out = hn.view(-1, self.hidden_units)
        out = self.relu(out)
        out = self.linear(out).flatten()
        # print(out)
        # print(x)

        return out

class ShallowMovementLSTM(torch.nn.Module):
    def __init__(self, sequence_length, num_features, hidden_units):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_features = num_features  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 2

        self.input = torch.nn.Linear(in_features=num_features, out_features=hidden_units)
        self.hidden1 = torch.nn.Linear(in_features=hidden_units, out_features=num_features)
        self.lstm = torch.nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )
        self.relu = torch.nn.ReLU()

        self.hidden2 = torch.nn.Linear(
            in_features=self.hidden_units, out_features=self.hidden_units)
        self.output = torch.nn.Linear(in_features=self.hidden_units, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.rand(self.num_layers, batch_size,
                         self.hidden_units, requires_grad=True)
        c0 = torch.rand(self.num_layers, batch_size,
                         self.hidden_units, requires_grad=True)

        x = self.input(x)
        x = self.relu(x)
        x = self.hidden1(x)
        x = self.relu(x)
        _, (hn, _) = self.lstm(x, (h0, c0))

        out = hn[-1].view(-1, self.hidden_units)
        out = self.hidden2(out)
        out = self.relu(out)
        out = self.output(out).flatten()
        out = self.sigmoid(out)
        # print(out)
        # print(x)

        return out
    
class ShallowLSTM(torch.nn.Module):
    def __init__(self, sequence_length, num_features, hidden_units):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_features = num_features  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 2

        self.input = torch.nn.Linear(in_features=num_features, out_features=hidden_units)
        self.hidden1 = torch.nn.Linear(in_features=hidden_units, out_features=num_features)
        self.lstm = torch.nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )
        self.relu = torch.nn.ReLU()

        self.hidden2 = torch.nn.Linear(
            in_features=self.hidden_units, out_features=self.hidden_units)
        self.output = torch.nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.rand(self.num_layers, batch_size,
                         self.hidden_units, requires_grad=True)
        c0 = torch.rand(self.num_layers, batch_size,
                         self.hidden_units, requires_grad=True)

        x = self.input(x)
        x = self.relu(x)
        x = self.hidden1(x)
        x = self.relu(x)
        _, (hn, _) = self.lstm(x, (h0, c0))

        out = hn[-1].view(-1, self.hidden_units)
        out = self.hidden2(out)
        out = self.relu(out)
        out = self.output(out).flatten()
        # print(out)
        # print(x)

        return out


class DoubleRegressionLSTM(torch.nn.Module):
    def __init__(self, sequence_length, num_features, hidden_units):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_features = num_features  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 2

        self.lstm = torch.nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )
        self.relu = torch.nn.ReLU()

        self.linear = torch.nn.Linear(
            in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))

        # Take the hidden units from the last layer
        out = hn[-1].view(-1, self.hidden_units)
        out = self.relu(out)
        out = self.linear(out).flatten()

        return out

class QuadRegressionLSTM(torch.nn.Module):
    def __init__(self, sequence_length, num_features, hidden_units):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_features = num_features  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 4

        self.lstm = torch.nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )
        self.relu = torch.nn.ReLU()

        self.linear = torch.nn.Linear(
            in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))

        # Take the hidden units from the last layer
        out = hn[-1].view(-1, self.hidden_units)
        out = self.relu(out)
        out = self.linear(out).flatten()

        return out

class DeepRegressionLSTM(torch.nn.Module):
    def __init__(self, sequence_length, num_features, hidden_units):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_features = num_features  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 16

        self.lstm = torch.nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )
        self.relu = torch.nn.ReLU()

        self.linear = torch.nn.Linear(
            in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))

        # Take the hidden units from the last layer
        out = hn[-1].view(-1, self.hidden_units)
        out = self.relu(out)
        out = self.linear(out).flatten()

        return out
