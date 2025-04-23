import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy.stats import ttest_ind
from dtw import *
from tabulate import tabulate
import os
import csv

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    plt.figure(figsize=(12, 6), dpi=300)
    plt.pcolormesh(x_edges, 
                   y_edges, 
                   histogram.T, 
                   shading='auto', 
                   cmap='viridis', 
                   norm=LogNorm(vmin=1, vmax=histogram.max()))
    plt.colorbar(label='Number of Occurrences (Log Scale)')
    plt.xlabel('Wavelengths', fontsize=18)
    plt.ylabel('Intensities', fontsize=18)
    plt.title(label, fontsize=18)
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.tight_layout()
    plt.show()


def create_difference_spectrogram(wavelengths,
                                  real_intensities,
                                  predicted_intensities,
                                  label="Difference Spectrogram"):
    """
    Create a spectrogram-like plot showing the differences between real and predicted intensities.

    Parameters:
        wavelengths (numpy.ndarray): 1D array of wavelength values.
        real_intensities (numpy.ndarray): 2D array of real intensities.
        predicted_intensities (numpy.ndarray): 2D array of predicted intensities.
        label (str): Title for the plot.

    Returns:
        None: Displays the spectrogram.
    """
    # Compute the differences
    intensity_differences = np.concatenate(real_intensities) - np.concatenate(predicted_intensities)

    # Flatten wavelengths for histogram
    all_wavelengths = np.tile(wavelengths, len(real_intensities))

    wavelength_bins = np.linspace(min(wavelengths), max(wavelengths), len(wavelengths) + 1)
    difference_bins = np.linspace(min(intensity_differences), max(intensity_differences), 100)

    histogram, x_edges, y_edges = np.histogram2d(all_wavelengths, intensity_differences,
                                                 bins=[wavelength_bins, difference_bins])

    plt.figure(figsize=(12, 6), dpi=300)
    plt.pcolormesh(x_edges,
                   y_edges,
                   histogram.T,
                   shading='auto',
                   cmap='seismic',
                   norm=LogNorm(vmin=1, vmax=histogram.max()))
    plt.colorbar(label='Number of Occurrences (Log Scale)')
    plt.xlabel('Wavelengths', fontsize=18)
    plt.ylabel('Differences (Real - Predicted)', fontsize=18)
    plt.title(label, fontsize=18)
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.ylim(-1200, 1200)
    
    plt.tight_layout()
    plt.show()


def create_combined_spectrogram(wavelengths, real, predicted):
    """
    Create a 2x2 grid of spectrograms: original, predicted, absolute error, and signed difference.

    Parameters:
        wavelengths (numpy.ndarray): 1D array of wavelength values.
        real (numpy.ndarray): 2D array of real intensities.
        predicted (numpy.ndarray): 2D array of predicted intensities.

    Returns:
        None: Displays the combined spectrogram.
    """
    # Flatten wavelengths and intensities
    all_wavelengths = np.tile(wavelengths, len(real))
    all_real = np.concatenate(real)
    all_predicted = np.concatenate(predicted)

    # Absolute differences for bottom-left
    abs_differences = np.abs(np.array(real) - np.array(predicted))
    all_abs_differences = np.concatenate(abs_differences)

    # Signed differences for bottom-right
    signed_differences = np.array(real) - np.array(predicted)
    all_signed_differences = np.concatenate(signed_differences)

    # Define bins for the histograms
    wavelength_bins = np.linspace(min(wavelengths), max(wavelengths), len(wavelengths) + 1)
    intensity_bins = np.linspace(min(all_real), max(all_real), 100)
    abs_difference_bins = np.linspace(min(all_abs_differences), max(all_abs_differences), 100)
    signed_difference_bins = np.linspace(min(all_signed_differences), max(all_signed_differences), 100)

    # Create histograms
    hist_real, _, _ = np.histogram2d(all_wavelengths, all_real, bins=[wavelength_bins, intensity_bins])
    hist_predicted, _, _ = np.histogram2d(all_wavelengths, all_predicted, bins=[wavelength_bins, intensity_bins])
    hist_abs_difference, _, _ = np.histogram2d(all_wavelengths, all_abs_differences,
                                               bins=[wavelength_bins, abs_difference_bins])
    hist_signed_difference, _, _ = np.histogram2d(all_wavelengths, all_signed_differences,
                                                  bins=[wavelength_bins, signed_difference_bins])

    # Plot the combined spectrogram
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=300)  # 2x2 grid

    # Top-left: Original spectrogram
    axes[0, 0].pcolormesh(wavelength_bins, intensity_bins, hist_real.T, shading='auto', cmap='viridis',
                          norm=LogNorm(vmin=1, vmax=hist_real.max()))
    axes[0, 0].set_title("Original Spectrogram")
    axes[0, 0].set_xlabel("Wavelengths")
    axes[0, 0].set_ylabel("Intensities")

    # Top-right: Predicted spectrogram
    axes[0, 1].pcolormesh(wavelength_bins, intensity_bins, hist_predicted.T, shading='auto', cmap='viridis',
                          norm=LogNorm(vmin=1, vmax=hist_predicted.max()))
    axes[0, 1].set_title("Predicted Spectrogram")
    axes[0, 1].set_xlabel("Wavelengths")
    axes[0, 1].set_ylabel("Intensities")

    # Bottom-left: Absolute difference spectrogram
    axes[1, 0].pcolormesh(wavelength_bins, abs_difference_bins, hist_abs_difference.T, shading='auto', cmap='viridis',
                          norm=LogNorm(vmin=1, vmax=hist_abs_difference.max()))
    axes[1, 0].set_title("Difference Spectrogram (Absolute Error)")
    axes[1, 0].set_xlabel("Wavelengths")
    axes[1, 0].set_ylabel("Absolute Intensity Differences")

    # Bottom-right: Signed difference spectrogram
    axes[1, 1].pcolormesh(wavelength_bins, signed_difference_bins, hist_signed_difference.T, shading='auto',
                          cmap='viridis',
                          norm=LogNorm(vmin=1, vmax=hist_signed_difference.max()))
    axes[1, 1].set_title("Difference Spectrogram (Signed: Real - Predicted)")
    axes[1, 1].set_xlabel("Wavelengths")
    axes[1, 1].set_ylabel("Signed Intensity Differences")

    plt.tight_layout()
    plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0], label="Occurrences (Log Scale)")
    plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1], label="Occurrences (Log Scale)")
    plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0], label="Occurrences (Log Scale)")
    plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label="Occurrences")

    plt.show()

def plot_random_intensity_vectors(wavelengths,
                                  predictions,
                                  cond_vectors,
                                  num_vectors=5):
    """
    Plot random predicted intensity vectors into one graph,
    including their corresponding conditional vectors in the legend.

    Parameters:
        wavelengths (numpy.ndarray): 1D array of wavelength values.
        predictions (numpy.ndarray): 2D array where each row is a vector of predicted intensities.
        cond_vectors (numpy.ndarray): 2D array where each row is the conditional vector corresponding to each predicted vector.
        num_vectors (int): Number of vectors to randomly select and plot.

    Returns:
        None: Displays the graph.
    """
    # Randomly select num_vectors indices from predictions
    random_indices = np.random.choice(len(predictions), size=num_vectors, replace=False)
    random_vectors = predictions[random_indices]
    random_cond_vectors = cond_vectors[random_indices]

    # Plot the selected vectors
    plt.figure(figsize=(12, 6))
    for i, (vector, cond_vector) in enumerate(zip(random_vectors, random_cond_vectors)):
        # Create the conditional vector string for the label
        cond_str = ', '.join([f"{val:.2f}" for val in cond_vector])
        plt.plot(wavelengths, vector, label=f"Vector {i+1} | Cond: [{cond_str}]")

    plt.title(f"{num_vectors} Random Predicted Intensity Vectors")
    plt.xlabel("Wavelengths")
    plt.ylabel("Intensities")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_combined_intensity_vectors_with_parameters(wavelengths,
                                                    real,
                                                    predicted,
                                                    cond_vectors,
                                                    indices,
                                                    num_vectors=10):
    """
    Plot a grid comparing true and predicted intensity vectors with conditional
    parameter values and MSE in titles.

    Parameters:
        wavelengths (numpy.ndarray): 1D array of wavelength values.
        real (numpy.ndarray): 2D array of true intensities.
        predicted (numpy.ndarray): 2D array of predicted intensities.
        cond_vectors (numpy.ndarray): 2D array where each row corresponds to conditional vector values.
        indices (numpy.ndarray): Indices of the selected vectors to plot.
        num_vectors (int): Number of vectors to plot (length of indices).

    Returns:
        None: Displays the comparison plot.
    """
    selected_real = real[indices]
    selected_predicted = predicted[indices]
    selected_conditions = cond_vectors[indices]

    fig, axes = plt.subplots(num_vectors, 1, figsize=(8, 4 * num_vectors), sharex=True, sharey=True, dpi=300)

    # If only one row (num_vectors=1), ensure axes is iterable
    if num_vectors == 1:
        axes = [axes]

    # Plot the vectors
    for i, ax in enumerate(axes):
        
        # Plot true and predicted vectors
        ax.plot(wavelengths, selected_real[i], label='True', color='tab:blue')
        ax.plot(wavelengths, selected_predicted[i], label='Predicted', color='tab:orange')
        
        cond_chunks = np.array_split(selected_conditions[i], 1) 
        cond_str = "\n".join([', '.join([f"{val:.2f}" for val in chunk]) for chunk in cond_chunks])

        ax.set_title(f"Conditional Vector:\n[{cond_str}]", fontsize=10)

        ax.set_ylim(0, 1200)

        if i == num_vectors - 1:
            ax.set_xlabel("Wavelengths")
        if i == 0:
            ax.set_ylabel("Intensity")
        ax.grid(True)
        

    axes[0].legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

    return selected_real, selected_predicted, selected_conditions


def get_worst_mse_indices(true_values,
                          predicted_values,
                          n=10):
    """
    Calculate MSE for each vector and return the indices of the vectors with the worst MSE.

    Parameters:
        true_values (numpy.ndarray): 2D array of true intensity values.
        predicted_values (numpy.ndarray): 2D array of predicted intensity values.
        n (int): Number of vectors with the worst MSE to select.

    Returns:
        numpy.ndarray: Indices of the vectors with the worst MSE.
    """
    mse = np.mean((true_values - predicted_values) ** 2, axis=1)

    # Get the indices of the n worst MSE values (highest MSE)
    worst_indices = np.argsort(mse)[-n:]

    return worst_indices

def get_best_mse_indices(true_values,
                          predicted_values,
                          n=10):
    """
    Calculate MSE for each vector and return the indices of the vectors with the best MSE.

    """
    mse = np.mean((true_values - predicted_values) ** 2, axis=1)

    # Get the indices of the n worst MSE values (highest MSE)
    best_indices = np.argsort(mse)[0:n]

    return best_indices

def mse_statistics(true_values, predicted_values):
    """
    Calculate and display statistical information about the MSE between true and predicted values,
    including boxplot statistics.

    Parameters:
        true_values (numpy.ndarray): 2D array of true intensity values.
        predicted_values (numpy.ndarray): 2D array of predicted intensity values.

    Returns:
        dict: A dictionary containing statistical measures of the MSE.
    """
    # Calculate the Mean Squared Error for each pair of vectors
    mse = np.mean((true_values - predicted_values) ** 2, axis=1)

    # Compute descriptive statistics
    descriptive_stats = {
        "Mean": np.mean(mse),
        "Median": np.median(mse),
        "Standard Deviation": np.std(mse),
        "Min": np.min(mse),
        "Max": np.max(mse),
        "Range": np.max(mse) - np.min(mse),
    }

    # Compute boxplot statistics
    Q1 = np.percentile(mse, 25)
    Q2 = np.median(mse)
    Q3 = np.percentile(mse, 75)
    IQR = Q3 - Q1
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR
    outliers = mse[(mse < lower_whisker) | (mse > upper_whisker)]

    boxplot_stats = {
        "Median (Q2)": Q2,
        "First Quartile (Q1)": Q1,
        "Third Quartile (Q3)": Q3,
        "Interquartile Range (IQR)": IQR,
        "Upper Whisker": upper_whisker,
        "Number of Outliers": len(outliers),
        "Outliers": outliers.tolist(),
    }

    all_stats = {**descriptive_stats, **boxplot_stats}

    print("MSE Statistics:")
    for key, value in all_stats.items():
        if isinstance(value, list):
            print(f"{key}: {value[:10]}{'...' if len(value) > 10 else ''}")
        else:
            print(f"{key}: {value:.4f}" if isinstance(value, (float, int)) else f"{key}: {value}")

    # Create boxplot of MSE values
    plt.figure(figsize=(12, 6), dpi=300)
    plt.boxplot(mse, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title("Boxplot of MSE Values")
    plt.xlabel("MSE")
    plt.grid(True)

    ticks = np.linspace(np.min(mse), np.max(mse), 10)
    plt.xticks(ticks, [f"{tick:.4f}" for tick in ticks])
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.hist(mse, bins=20, color='skyblue', edgecolor='black')
    plt.title("Histogram of MSE Values")
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    return all_stats


def analyze_mse_and_cond_vectors(mse,
                                 cond_vectors,
                                 parameter_names):
    """
    Analyzes the MSE and conditional vectors, comparing high MSE vs low MSE,
    and includes parameter names in the boxplot titles.

    Parameters:
        mse (numpy.ndarray): Array of MSE values.
        cond_vectors (numpy.ndarray): Conditional vectors corresponding to predicted vectors.
        parameter_names (list): List of strings representing the names of the
        parameters in conditional vectors.

    Returns:
        None
    """
    Q1 = np.percentile(mse, 25)
    Q3 = np.percentile(mse, 75)
    IQR = Q3 - Q1
    upper_whisker = Q3 + 1.5 * IQR

    high_mse_indices = np.where(mse > upper_whisker)[0]
    low_mse_indices = np.where(mse <= upper_whisker)[0]

    high_mse_cond_vectors = cond_vectors[high_mse_indices]
    low_mse_cond_vectors = cond_vectors[low_mse_indices]

    print(f"\nNumber of vectors with high MSE (above upper whisker): {high_mse_cond_vectors.shape[0]}")
    print(f"Number of vectors with low MSE (below or equal to upper whisker): {low_mse_cond_vectors.shape[0]}")

    # Function to calculate basic statistics for conditional vectors
    def calculate_statistics(vectors):
        stats = {
            'mean': np.mean(vectors, axis=0),
            'median': np.median(vectors, axis=0),
            'std': np.std(vectors, axis=0),
            'min': np.min(vectors, axis=0),
            'max': np.max(vectors, axis=0)
        }
        return stats

    # Calculate statistics for both high and low MSE conditional vectors
    high_mse_stats = calculate_statistics(high_mse_cond_vectors)
    low_mse_stats = calculate_statistics(low_mse_cond_vectors)

    print("\nStatistics for high MSE conditional vectors:")
    for stat, value in high_mse_stats.items():
        print(f"  {stat.capitalize()}: {value}")

    print("\nStatistics for low MSE conditional vectors:")
    for stat, value in low_mse_stats.items():
        print(f"  {stat.capitalize()}: {value}")

    fig, axs = plt.subplots(3, 1, figsize=(6, 12), dpi=300)

    for i, param_name in enumerate(parameter_names):
        ax = axs[i]
        high_mse_param = high_mse_cond_vectors[:, i]
        low_mse_param = low_mse_cond_vectors[:, i]

        df_high = pd.DataFrame({'Value': high_mse_param, 'MSE Category': ['High MSE'] * len(high_mse_param)})
        df_low = pd.DataFrame({'Value': low_mse_param, 'MSE Category': ['Low MSE'] * len(low_mse_param)})

        df_combined = pd.concat([df_high, df_low], ignore_index=True)

        sns.boxplot(x='MSE Category', y='Value', data=df_combined, ax=ax)
        ax.set_title(f'{param_name} Distribution: High vs Low MSE')
        ax.set_ylabel(f'{param_name} Values')

    plt.tight_layout()
    plt.show()


def plot_predictions_with_cond_vectors(predictions, cond_vectors, num_rows=20):
    """
    Plots predicted intensity vectors and displays corresponding conditional vectors split into multiple rows.

    Parameters:
        predictions (numpy.ndarray): Array of predicted intensity vectors.
        cond_vectors (numpy.ndarray): Array of corresponding conditional vectors.
        num_rows (int): Number of rows in the subgraph.

    Returns:
        None
    """
    fig, axes = plt.subplots(num_rows, 1, figsize=(7, 45), dpi=300)

    for i in range(num_rows):
        ax = axes[i]
        ax.plot(predictions[i], label='Predicted Intensities', color='b')

        cond_chunks = np.array_split(cond_vectors[i], 3) 
        cond_str = "\n".join([', '.join([f"{val:.2f}" for val in chunk]) for chunk in cond_chunks])

        ax.set_title(f"Conditional Vector:\n[{cond_str}]", fontsize=10)
        ax.set_xlabel('Index')
        ax.set_ylabel('Intensity')
        ax.grid(True)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

#def calculate_metrics(wavelengths, intensities):
#    """Calculate mean wavelength, FWHM, and FWTM."""
#    intensities_changed = intensities.copy()
#    intensities_changed += abs(min(intensities))
#    mean_wavelength = np.sum(wavelengths * intensities_changed) / np.sum(intensities_changed)
#    std_deviation = np.sqrt(np.sum(intensities_changed * (wavelengths - mean_wavelength) ** 2) / np.sum(intensities_changed))
    
#    half_max = (max(intensities_changed) + min(intensities_changed)) / 2
#    indices_above_half_max = np.where(intensities_changed >= half_max)[0]
#    fwhm_start = wavelengths[indices_above_half_max[0]]
#    fwhm_end = wavelengths[indices_above_half_max[-1]]
#    fwhm = fwhm_end - fwhm_start

#    tenth_max = (max(intensities_changed) + min(intensities_changed)) / 10
#    indices_above_tenth_max = np.where(intensities_changed >= tenth_max)[0]
#    fwtm_start = wavelengths[indices_above_tenth_max[0]]
#    fwtm_end = wavelengths[indices_above_tenth_max[-1]]
#    fwtm = fwtm_end - fwtm_start

#    return mean_wavelength, std_deviation, fwhm, fwhm_start, fwhm_end, fwtm, fwtm_start, fwtm_end

def calculate_metrics(wavelengths, intensities):
    """Calculate mean wavelength, FWHM, and FWTM."""
    sum_int = np.sum(intensities)
    if sum_int == 0:
        sum_int = 1e-8
    mean_wavelength = np.sum(wavelengths * intensities) / sum_int
    std_deviation = np.sqrt(np.sum(intensities * (wavelengths - mean_wavelength) ** 2) / sum_int)
    
    half_max = (max(intensities) + min(intensities)) / 2
    indices_above_half_max = np.where(intensities >= half_max)[0]
    fwhm_start = wavelengths[indices_above_half_max[0]]
    fwhm_end = wavelengths[indices_above_half_max[-1]]
    fwhm = fwhm_end - fwhm_start

    tenth_max = (max(intensities) + min(intensities)) / 10
    indices_above_tenth_max = np.where(intensities >= tenth_max)[0]
    fwtm_start = wavelengths[indices_above_tenth_max[0]]
    fwtm_end = wavelengths[indices_above_tenth_max[-1]]
    fwtm = fwtm_end - fwtm_start

    return mean_wavelength, std_deviation, fwhm, fwhm_start, fwhm_end, fwtm, fwtm_start, fwtm_end

def calculate_metrics_errors(wavelengths, x_real, predictions):
    """Calculate metrics for each pair of vectors and compute MSE and MAE for each metric."""
    num_samples = x_real.shape[0]

    # Initialize lists to store metrics
    metrics_real = []
    metrics_pred = []

    # Calculate metrics for each pair of vectors
    for i in range(num_samples):
        metrics_real.append(calculate_metrics(wavelengths, x_real[i]))
        metrics_pred.append(calculate_metrics(wavelengths, predictions[i]))

    # Convert to numpy arrays for easier manipulation
    metrics_real = np.array(metrics_real)
    metrics_pred = np.array(metrics_pred)

    # Calculate MSE and MAE for each metric
    mse_results = {}
    mae_results = {}
    metric_names = ["mean_wavelength", "std_deviation", "fwhm", "fwhm_start", "fwhm_end", "fwtm", "fwtm_start", "fwtm_end"]

    for j, name in enumerate(metric_names):
        mse = np.mean((metrics_real[:, j] - metrics_pred[:, j]) ** 2)
        mae = np.mean(np.abs(metrics_real[:, j] - metrics_pred[:, j]))
        mse_results[name] = mse
        mae_results[name] = mae

    # Print results in a table
    table_data = [[metric, f"{mse_results[metric]:.6f}", f"{mae_results[metric]:.6f}"] for metric in metric_names]
    print(tabulate(table_data, headers=["Metric", "MSE", "MAE"], tablefmt="grid"))

    return mse_results, mae_results

def plot_comparison(index, wavelengths, x_real, predictions, cond_vectors):
    """
    Plot the intensity vectors for the given index from x_real and predictions 
    with their corresponding FWHM, FWTM, and center of gravity in subplots.

    Parameters:
        index (int): Index of the intensity vector to plot.
        wavelengths (numpy.ndarray): 1D array of wavelength values.
        x_real (numpy.ndarray): 2D array where each row is a vector of real intensities.
        predictions (numpy.ndarray): 2D array where each row is a vector of predicted intensities.

    Returns:
        None: Displays the plot with subplots.
    """
    

    # Extract the real and predicted intensity vectors for the given index
    real_intensities = x_real[index]
    predicted_intensities = predictions[index]

    # Calculate metrics for real and predicted intensity vectors
    real_metrics = calculate_metrics(wavelengths, real_intensities)
    pred_metrics = calculate_metrics(wavelengths, predicted_intensities)

    # Set up the subplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot the real intensity vector
    axes[0].plot(wavelengths, real_intensities, label='Real Intensity')
    axes[0].axvline(real_metrics[0], color='red', linestyle='--', label=f'Mean Wavelength: {real_metrics[0]:.2f} nm')
    axes[0].axvline(real_metrics[3], color='green', linestyle=':', label=f'FWHM Start: {real_metrics[3]:.2f} nm')
    axes[0].axvline(real_metrics[4], color='green', linestyle=':', label=f'FWHM End: {real_metrics[4]:.2f} nm')
    axes[0].fill_betweenx([min(real_intensities), 
                           max(real_intensities)], 
                          real_metrics[3], 
                          real_metrics[4],
                          color='green', 
                          alpha=0.1, 
                          label=f'FWHM: {real_metrics[2]:.2f} nm')
    axes[0].axvline(real_metrics[6], color='blue', linestyle=':', label=f'FWTM Start: {real_metrics[6]:.2f} nm')
    axes[0].axvline(real_metrics[7], color='blue', linestyle=':', label=f'FWTM End: {real_metrics[7]:.2f} nm')
    axes[0].fill_betweenx([min(real_intensities), 
                           max(real_intensities)], 
                          real_metrics[6], 
                          real_metrics[7], 
                          color='blue', 
                          alpha=0.1, 
                          label=f'FWTM: {real_metrics[5]:.2f} nm')
    axes[0].set_title('Real Intensity')
    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('Intensity')
    axes[0].legend()
    axes[0].grid(True)

    # Plot the predicted intensity vector
    axes[1].plot(wavelengths, predicted_intensities, label='Predicted Intensity')
    axes[1].axvline(pred_metrics[0], color='red', linestyle='--', label=f'Mean Wavelength: {pred_metrics[0]:.2f} nm')
    axes[1].axvline(pred_metrics[3], color='green', linestyle=':', label=f'FWHM Start: {pred_metrics[3]:.2f} nm')
    axes[1].axvline(pred_metrics[4], color='green', linestyle=':', label=f'FWHM End: {pred_metrics[4]:.2f} nm')
    axes[1].fill_betweenx([min(predicted_intensities), 
                           max(predicted_intensities)], 
                          pred_metrics[3], 
                          pred_metrics[4], 
                          color='green', 
                          alpha=0.1, 
                          label=f'FWHM: {pred_metrics[2]:.2f} nm')
    axes[1].axvline(pred_metrics[6], color='blue', linestyle=':', label=f'FWTM Start: {pred_metrics[6]:.2f} nm')
    axes[1].axvline(pred_metrics[7], color='blue', linestyle=':', label=f'FWTM End: {pred_metrics[7]:.2f} nm')
    axes[1].fill_betweenx([min(predicted_intensities), 
                           max(predicted_intensities)], 
                          pred_metrics[6], 
                          pred_metrics[7], 
                          color='blue', 
                          alpha=0.1, 
                          label=f'FWTM: {pred_metrics[5]:.2f} nm')
    axes[1].set_title('Predicted Intensity')
    axes[1].set_xlabel('Wavelength (nm)')
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(f'Index: {index}, Conditional vector: {cond_vectors[index]}')

    plt.tight_layout()
    plt.show()



