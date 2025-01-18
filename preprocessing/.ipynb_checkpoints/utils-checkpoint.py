import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy.stats import ttest_ind
import os
import csv

def plot_3d_spectrogram(df, 
                        specific_idx, 
                        time_col='Time', 
                        wavelengths_col='Wavelengths',
                        intensities_col = 'Intensities'
                       ):
    """
    Creates a 3D spectrogram for a specific idx from the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing 'Time', 
    'Intensities', 'Wavelengths', and 'idx' columns.
    - specific_idx (int): The specific idx to filter the DataFrame.
    
    Returns:
    - None: Displays the 3D spectrogram plot.
    """
    # Filter the DataFrame based on the specific idx
    filtered_df = df[df['idx'] == specific_idx]
        
    if filtered_df.empty:
        print(f"No data found for idx: {specific_idx}")
        return
    
    # Prepare data for plotting
    times = pd.to_datetime(filtered_df[time_col])
    wavelengths = np.array(filtered_df[wavelengths_col].iloc[0])
    intensities = np.array([intensity for intensity in filtered_df[intensities_col]])
    
    # Create meshgrid for the 3D plot
    T, W = np.meshgrid(times.astype(np.int64) / 1e9, wavelengths)
    
    # Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(W, T, intensities.T, cmap='viridis')

    # Set plot labels (SWITCHED)
    ax.set_xlabel("Wavelength (nm)")  
    ax.set_ylabel("Time (s)")        
    ax.set_zlabel("Intensity")
    ax.set_title(f"3D Spectrogram for idx: {specific_idx}, number of datapoints: {len(filtered_df)}")
    
    # Add color bar for intensity scale
    fig.colorbar(surf, ax=ax, label="Intensity")

    plt.show()
    
def spec3d_freq_ind(df, 
                    threshold, 
                    time_col='Time', 
                    wavelengths_col='Wavelengths',
                    intensities_col = 'Intensities'
                   ):
    # Get counts of each idx
    idx_counts = df['idx'].value_counts()
    
    # Filter to get idx values with count > threshold
    frequent_indices = idx_counts[idx_counts > threshold].index
    
    # Plot for each frequent idx
    for specific_idx in frequent_indices:
        plot_3d_spectrogram(df, 
                            specific_idx,
                            time_col=time_col, 
                            wavelengths_col=wavelengths_col,
                            intensities_col=intensities_col
                           )

def create_spectrogram(wavelengths,
                       intensities,
                       label="Spectrogram of Intensities by Wavelength"):
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

def print_feature_statistics(data):
    # List of columns to analyze
    features = [
        "Stage3_OutputPower",
        "Stage3_Piezo",
        "Stage3_Stepper",
        "stepper_diff"
    ]
    
    for feature in features:
        if feature in data.columns:
            # Calculate statistics
            min_val = np.min(data[feature])
            max_val = np.max(data[feature])
            mean_val = np.mean(data[feature])
            median_val = np.median(data[feature])
            variance_val = np.var(data[feature])

            # Print statistics
            print(f"\nStatistics for {feature}:")
            print(f"  Minimum: {min_val}")
            print(f"  Maximum: {max_val}")
            print(f"  Mean: {mean_val}")
            print(f"  Median: {median_val}")
            print(f"  Variance: {variance_val}")
        else:
            print(f"\nFeature {feature} not found in the DataFrame")

def custom_bin_histogram(data_array, label, bins=None):
    
    if bins is None:
        bins = [-np.inf] + list(np.arange(-20, 4000, 50)) + [np.inf]

    # Generate labels for bins
    bin_labels = [f"<{bins[i]},{bins[i+1]})" for i in range(len(bins)-1)]

    # Bin the data
    binned_counts, _ = np.histogram(data_array, bins=bins)

    print("Bin counts:")
    for i, count in enumerate(binned_counts):
        print(f"  {bin_labels[i]}: {count} values", end=", ")

    plt.figure(figsize=(12, 6))
    plt.bar(bin_labels, 
            binned_counts, 
            color='blue', 
            alpha=0.7, 
            edgecolor='black')
    
    plt.title(label)
    plt.xlabel("Value Range")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()

def analyze_outliers(data, lower_percentile, upper_percentile):
    """
    Analyzes data for outliers based on specified percentile bounds and generates a histogram.
    
    Parameters:
        data (array-like): The data to analyze.
        lower_percentile (float): The lower percentile for calculating bounds (e.g., 0.1).
        upper_percentile (float): The upper percentile for calculating bounds (e.g., 99.9).
    
    Returns:
        tuple: A tuple containing the lower bound and upper bound.
    """
    # Calculate lower and upper bounds
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)

    print(f"Typical range: {lower_bound} to {upper_bound}")

    # Identify outliers
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    print(f"Outliers: {outliers}")

    # Plot histogram with bounds
    plt.hist(data, bins=50, alpha=0.7, label="Original Data")
    plt.axvline(lower_bound, color='r', linestyle='--', label=f"{lower_percentile}th Percentile")
    plt.axvline(upper_bound, color='g', linestyle='--', label=f"{upper_percentile}th Percentile")
    plt.legend()
    plt.title("Data Distribution with Outlier Bounds")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.show()

    return lower_bound, upper_bound

def clip_and_normalize(data, lower_bound, upper_bound, norm_min=0, norm_max=1):
    """
    Clips and normalizes the data to a specified range.
    
    Args:
        data: The input data to be processed (e.g., numpy array or list).
        lower_bound: The lower bound for clipping.
        upper_bound: The upper bound for clipping.
        norm_min: The minimum value of the normalization range. Default is -1.
        norm_max: The maximum value of the normalization range. Default is 1.
    
    Returns:
        Normalized data within the specified range.
    """
    # Clip the data
    clipped_data = np.clip(data, lower_bound, upper_bound)
    
    # Normalize the clipped data to the desired range
    normalized_data = norm_min + (clipped_data - lower_bound) * (norm_max - norm_min) / (upper_bound - lower_bound)
    
    return normalized_data

def denormalize(data, lower_bound, upper_bound, norm_min=-1, norm_max=1):
    """
    Denormalizes data from a specified range back to its original range.
    
    Args:
        data: The normalized data to be transformed (e.g., numpy array or list).
        lower_bound: The original lower bound of the data range.
        upper_bound: The original upper bound of the data range.
        norm_min: The minimum value of the normalization range. Default is -1.
        norm_max: The maximum value of the normalization range. Default is 1.
    
    Returns:
        Data transformed back to the original range.
    """
    # Reverse the normalization
    original_data = lower_bound + (data - norm_min) * (upper_bound - lower_bound) / (norm_max - norm_min)
    
    return original_data


