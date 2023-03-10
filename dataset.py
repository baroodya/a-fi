import torch


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, features, target, sequence_length, sequence_sep=1):
        self.sequence_length = sequence_length
        self.sequence_sep = sequence_sep
        self.X = torch.tensor(dataframe[features].values, dtype=torch.float32)
        self.y = torch.tensor(dataframe[target].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    # i is the index of the day I want to predict
    def __getitem__(self, i):
        if i >= self.sequence_length - self.sequence_sep + 1:
            i_start = i - self.sequence_length - self.sequence_sep + 1
            x = self.X[i_start:(i - self.sequence_sep + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i, 1)
            x = self.X[0:(i - self.sequence_sep + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]
