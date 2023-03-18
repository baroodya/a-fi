import torch


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, features, target, sequence_length, sequence_sep):
        self.sequence_length = sequence_length
        self.sequence_sep = sequence_sep
        self.X = torch.tensor(dataframe[features].values, dtype=torch.float32)
        self.y = torch.tensor(dataframe[target].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    # i is the index of the day I want to predict
    def __getitem__(self, i):
        if i >= self.sequence_length + self.sequence_sep:
            i_start = i - self.sequence_length - self.sequence_sep
            x = self.X[i_start:(i - self.sequence_sep), :]
        else:
            padding_length = self.sequence_length + self.sequence_sep - i
            if i < self.sequence_sep:
                padding_length -= 1
            padding = self.X[0].repeat(padding_length, 1)
            x = self.X[0:max(i - self.sequence_sep, 0), :]
            x = torch.cat((padding, x), 0)
        return x, self.y[i]
