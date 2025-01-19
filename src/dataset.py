import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data  # 1D feature vectors
        self.y_data = y_data  # Target labels

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        #x_tensor = torch.tensor(self.x_data[idx], dtype=torch.float32).reshape(1,-1)# convert x_data to tensor, reshape to add dimension
        x_tensor = torch.tensor(self.x_data[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y_data[idx], dtype=torch.float32)  # convert y_data to tensor
        return {'data': x_tensor, 'settings': y_tensor}
        #return {'data': self.x_data[idx], 'settings': self.y_data[idx]}

def get_data(data_path):
    df = pd.read_csv(data_path)
    df['intensities'] = df['intensities'].apply(eval)
    df['cond_vector'] = df['cond_vector'].apply(eval)

    x_data = df['intensities'].tolist()
    y_data = df['cond_vector'].tolist()

    return x_data, y_data