import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data  
        self.y_data = y_data  

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x_tensor = torch.tensor(self.x_data[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y_data[idx], dtype=torch.float32) 
        return {'data': x_tensor, 'settings': y_tensor}

def get_data(data_path):
    df = pd.read_csv(data_path)
    df['intensities'] = df['intensities'].apply(eval)
    df['cond_vector'] = df['cond_vector'].apply(eval)

    x_data = df['intensities'].tolist()
    y_data = df['cond_vector'].tolist()

    return x_data, y_data