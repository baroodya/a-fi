import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class FeatureDataset(Dataset):
    def __init__(self, file_name, scale=False):
        # read csv file and load row data into variables
        file_out = pd.read_csv(file_name)
        num_features = file_out.shape[1] - 1

        x = file_out.iloc[:-1, :num_features].to_numpy(dtype=float)
        y = file_out.iloc[:-1, num_features].values

        # Feature Scaling
        if scale:
            sc = StandardScaler()
            x_train = sc.fit_transform(x_train)
            x_test = sc.fit_transform(x_test)

        # Converting to torch tensors
        self.X = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
