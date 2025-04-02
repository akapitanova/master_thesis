import numpy as np
from scipy.stats import wasserstein_distance
from dtw import *

def calculate_dtw_distances(x_real, predictions):
    """Calculate Dynamic Time Warping (DTW) distance for each pair of vectors."""
    num_samples = x_real.shape[0]
    dtw_distances = []

    for i in range(num_samples):
        alignment = dtw(x_real[i], predictions[i], distance_only=True)
        dtw_distances.append(alignment.distance)

    return dtw_distances

def calculate_dtw_for_index(x_real, predictions, index):
    """Calculate Dynamic Time Warping (DTW) distance for a specific pair of vectors at the given index."""
    if index >= len(x_real) or index >= len(predictions):
        raise ValueError("Index out of bounds for the provided arrays.")

    alignment = dtw(x_real[index], predictions[index], keep_internals=True)
    print(f"DTW Distance for index {index}: {alignment.distance:.6f}")

    return alignment

def calculate_wasserstein_distance(x_real, predictions):
    """
    Compute the Wasserstein distance between two sets of distributions.
    
    Parameters:
    - x_real: np.array representing real data.
    - predictions: np.array representing generated data.
    
    Returns:
    - wasserstein_distances: Array of Wasserstein distances for each feature (column).
    - avg_distance: Average Wasserstein distance across all features.
    """
    num_features = x_real.shape[1]
    wasserstein_distances = np.zeros(num_features)

    for i in range(num_features):
        wasserstein_distances[i] = wasserstein_distance(x_real[:, i], predictions[:, i])

    avg_distance = np.mean(wasserstein_distances)
    return wasserstein_distances, avg_distance

def plot_dtw_alignment_for_index(x_real, predictions, index, plot_type="threeway"):
    alignment = calculate_dtw_for_index(x_real, predictions, index)
    alignment.plot(type=plot_type)
    



