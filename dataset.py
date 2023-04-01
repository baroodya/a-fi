import torch


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, features, target, sequence_length, sequence_sep):
        self.sequence_length = sequence_length
        self.sequence_sep = sequence_sep
        self.X = torch.tensor(dataframe[features].values, dtype=torch.float32)
        self.y = torch.tensor(dataframe[target].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y) - self.sequence_length - max(0, self.sequence_sep)

    # i is the index of the day I want to predict
    def __getitem__(self, i):
        x = self.X[i:i + self.sequence_length]
        y = self.y[i+self.sequence_length+self.sequence_sep-1]
        return x, y