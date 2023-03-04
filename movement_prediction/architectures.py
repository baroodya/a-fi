import torch


def createBasicModel(input_size):
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
        torch.nn.Sigmoid(),
        torch.nn.Flatten(0, 1),
    )


def createDeepModel(input_size):
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
        torch.nn.Sigmoid(),
        torch.nn.Flatten(0, 1),
    )


class MovementShallowRegressionLSTM(torch.nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = torch.nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = torch.nn.Linear(
            in_features=self.hidden_units, out_features=1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        # First dim of Hn is num_layers, which is set to 1 above.
        out = self.linear(hn[0]).flatten()
        out = self.sigmoid(out)

        return out
