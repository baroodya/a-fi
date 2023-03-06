import torch


class ShallowRegressionLSTM(torch.nn.Module):
    def __init__(self, num_features, hidden_units):
        super().__init__()
        self.num_sensors = num_features  # this is the number of features
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

        return out
