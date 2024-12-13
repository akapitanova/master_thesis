import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from torch.utils.data import DataLoader, Dataset


def create_spectrogram(wavelengths, intensities, label="Spectrogram of Intensities by Wavelength"):
    """
    Create and display a spectrogram-like plot.

    Parameters:
        wavelengths (numpy.ndarray): 1D array of wavelength values.
        intensities (numpy.ndarray): 2D array where each row corresponds to intensities for a wavelength.

    Returns:
        None: Displays the spectrogram.
    """
    # Flatten the wavelengths and intensities into paired values
    all_wavelengths = np.tile(wavelengths, len(intensities))
    all_intensities = np.concatenate(intensities)

    # Define bins for the 2D histogram
    wavelength_bins = np.linspace(min(wavelengths), max(wavelengths), len(wavelengths) + 1)
    intensity_bins = np.linspace(min(all_intensities), max(all_intensities), 100)

    # Create 2D histogram
    histogram, x_edges, y_edges = np.histogram2d(all_wavelengths, all_intensities, bins=[wavelength_bins, intensity_bins])

    # Plot the spectrogram with a logarithmic color scale
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(x_edges, 
                   y_edges, 
                   histogram.T, 
                   shading='auto', 
                   cmap='viridis', 
                   norm=LogNorm(vmin=1, vmax=histogram.max()))
    plt.colorbar(label='Number of Occurrences (Log Scale)')
    plt.xlabel('Wavelengths')
    plt.ylabel('Intensities')
    plt.title(label)
    
    plt.tight_layout()
    plt.show()


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