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

def clip_and_normalize(data, lower_bound, upper_bound, norm_min=-1, norm_max=1):
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

def plot_filtered_max_intensities_histogram(df, column="Intensities", bins=100, threshold=None):
    """
    Plots a histogram of the maximal values from the NumPy arrays in the specified column,
    but only for rows where the max value is below the given threshold.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column containing NumPy arrays.
        bins (int): Number of bins for the histogram.
        threshold (float): The upper limit for max values to be included.
    """
    # Extract max values from each row's "Intensities" array
    max_values = df[column].apply(lambda x: np.max(x))

    if threshold:
        # Filter values below the threshold
        filtered_values = max_values[max_values < threshold]
    else: 
        filtered_values = max_values

    # Check if there are values to plot
    if filtered_values.empty:
        print("No values below the threshold to plot.")
        return

    # Plot the histogram
    plt.figure(figsize=(8, 5))
    plt.hist(filtered_values, bins=bins, edgecolor="black", alpha=0.7)
    plt.xlabel("Maximum Intensity")
    plt.ylabel("Frequency")
    if threshold:
        plt.title(f"Histogram of Max Intensities (< {threshold})")
    else:
        plt.title(f"Histogram of Max Intensities")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def plot_random_intensity_vectors(wavelengths,
                                  intensities,
                                  num_vectors=5):
    """
    Plot random predicted intensity vectors into one graph.

    Parameters:
        wavelengths (numpy.ndarray): 1D array of wavelength values.
        intensities (numpy.ndarray): 2D array where each row is a vector of intensities.
        num_vectors (int): Number of vectors to randomly select and plot.

    Returns:
        None: Displays the graph.
    """
    # Randomly select num_vectors indices from predictions
    random_indices = np.random.choice(len(intensities), size=num_vectors, replace=False)
    print(random_indices)
    random_vectors = intensities[random_indices]

    # Plot the selected vectors
    plt.figure(figsize=(12, 6))
    for i, vector in enumerate(random_vectors):
        plt.plot(wavelengths, vector, label=f"Vector {i+1}")

    plt.title(f"{num_vectors} Random Intensity Vectors")
    plt.xlabel("Wavelengths")
    plt.ylabel("Intensities")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_hourly_histogram(df, time_column='Time'):
    """
    Plots a histogram showing the count of instances per hour.

    Parameters:
        df (pd.DataFrame): DataFrame containing a datetime column.
        time_column (str): Column name containing datetime values.

    Returns:
        None: Displays the histogram.
    """
    df[time_column] = pd.to_datetime(df[time_column])
    df['Hour'] = df[time_column].dt.hour
    hourly_counts = df['Hour'].value_counts().sort_index()

    min_hour, max_hour = hourly_counts.index.min(), hourly_counts.index.max()

    # Ensure all hours within this range are present (fill missing with 0)
    full_range = np.arange(min_hour, max_hour + 1)
    hourly_counts = hourly_counts.reindex(full_range, fill_value=0)

    plt.figure(figsize=(12, 6))
    hourly_counts.plot(kind='bar', color='skyblue', width=0.8)
    plt.title('Hourly Distribution of Entries (Daily Harmonogram)')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Entries')
    plt.xticks(range(len(full_range)), labels=[f"{h}:00" for h in full_range], rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def plot_daily_hourly_histograms(df, time_column='Time'):
    """
    Plots hourly histograms for each unique day in the dataset, sorted chronologically.

    Parameters:
        df (pd.DataFrame): DataFrame containing a datetime column.
        time_column (str): Column name containing datetime values.

    Returns:
        None: Displays multiple histograms (one per day).
    """
    df[time_column] = pd.to_datetime(df[time_column])  
    df['Date'] = df[time_column].dt.date 
    df['Hour'] = df[time_column].dt.hour 

    unique_dates = sorted(df['Date'].unique())

    for date in unique_dates:
        daily_data = df[df['Date'] == date]
        hourly_counts = daily_data['Hour'].value_counts().sort_index()

        # Ensure all hours are represented (fill missing with 0)
        min_hour, max_hour = hourly_counts.index.min(), hourly_counts.index.max()
        full_range = np.arange(min_hour, max_hour + 1)
        hourly_counts = hourly_counts.reindex(full_range, fill_value=0)

        plt.figure(figsize=(12, 6))
        hourly_counts.plot(kind='bar', color='skyblue', width=0.8)

        plt.title(f'Hourly Distribution of Entries on {date}')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Number of Entries')
        plt.xticks(range(len(full_range)), labels=[f"{h}:00" for h in full_range], rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

def plot_daily_max_intensity(df, time_column='Time', intensity_column='Intensities'):
    """
    Plots the maximum intensity per hour for each unique day in the dataset, sorted chronologically.

    Parameters:
        df (pd.DataFrame): DataFrame containing a datetime column and intensity values.
        time_column (str): Column name containing datetime values.
        intensity_column (str): Column name containing intensity values (lists or arrays).

    Returns:
        None: Displays multiple line plots (one per day).
    """
    df[time_column] = pd.to_datetime(df[time_column])
    df['Date'] = df[time_column].dt.date
    df['Hour'] = df[time_column].dt.hour

    unique_dates = sorted(df['Date'].unique())

    for date in unique_dates:
        daily_data = df[df['Date'] == date].copy()
        daily_data[intensity_column] = daily_data[intensity_column].apply(lambda x: np.max(np.array(x)) if isinstance(x, (list, np.ndarray)) else np.nan)
        hourly_max_intensity = daily_data.groupby('Hour')[intensity_column].max()

        min_hour, max_hour = hourly_max_intensity.index.min(), hourly_max_intensity.index.max()
        full_range = np.arange(min_hour, max_hour + 1)
        hourly_max_intensity = hourly_max_intensity.reindex(full_range, fill_value=np.nan)

        # Bar plot for maximum intensity values per hour
        plt.figure(figsize=(12, 6))
        hourly_max_intensity.plot(kind='bar', color='skyblue', edgecolor='black', width=0.8)
        plt.title(f'Maximum Intensity per Hour on {date}')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Max Intensity')
        plt.xticks(range(len(full_range)), labels=[f"{h}:00" for h in full_range], rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()

def calculate_fwtm(wavelengths, intensities):
    max_intensity = np.max(intensities)
    tenth_max = max_intensity / 10
    left_index = np.where(intensities >= tenth_max)[0][0]
    right_index = np.where(intensities >= tenth_max)[0][-1]
    fwtm = wavelengths[right_index] - wavelengths[left_index]
    return fwtm, wavelengths[left_index], wavelengths[right_index]

def plot_mean_intensity_per_hour_for_day(df, 
                                         specific_date, 
                                         time_column='Time', 
                                         intensity_column='Intensities', 
                                         wavelengths=None):
    df[time_column] = pd.to_datetime(df[time_column])
    df['Date'] = df[time_column].dt.date
    daily_data = df[df['Date'] == specific_date]
    daily_data['Hour'] = daily_data[time_column].dt.hour
    unique_hours = sorted(daily_data['Hour'].unique())

    for hour in unique_hours:
        hourly_data = daily_data[daily_data['Hour'] == hour]
        intensity_vectors = hourly_data[intensity_column].apply(lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) else np.nan).dropna()

        if len(intensity_vectors) > 0:
            mean_intensity_vector = np.mean(np.vstack(intensity_vectors), axis=0)

            mean_wavelength = np.sum(wavelengths * mean_intensity_vector) / np.sum(mean_intensity_vector)
            fwtm, fwtm_start, fwtm_end = calculate_fwtm(wavelengths, mean_intensity_vector)

            # Plotting
            plt.figure(figsize=(12, 6))
            plt.plot(wavelengths, mean_intensity_vector, label=f'Mean Intensity at Hour {hour}')
            plt.axvline(mean_wavelength, color='red', linestyle='--', label=f'Mean Wavelength: {mean_wavelength:.2f} nm')
            plt.axvline(fwtm_start, color='blue', linestyle=':', label=f'FWTM Start: {fwtm_start:.2f} nm')
            plt.axvline(fwtm_end, color='blue', linestyle=':', label=f'FWTM End: {fwtm_end:.2f} nm')
            plt.fill_betweenx([min(mean_intensity_vector), max(mean_intensity_vector)], fwtm_start, fwtm_end, color='blue', alpha=0.1, label=f'FWTM: {fwtm:.2f} nm')

            plt.title(f'Mean Intensity Vector for Hour {hour} on {specific_date}')
            plt.xlabel('Wavelengths')
            plt.ylabel('Mean Intensity')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()