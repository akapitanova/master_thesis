import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from evaluate import predict, save_predictions
from metrics import calculate_dtw_distances, calculate_wasserstein_distance


def setup_logging(run_name):
    """Creates necessary directories for model training and results storage."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    os.makedirs(os.path.join("predictions", run_name), exist_ok=True)


def save_samples(epoch, 
                 sampler, 
                 wavelengths,
                 args,
                 s_type='edm',
                 model=None
                ):
    """Generates and saves samples using the trained model."""

    if s_type == 'ddim':
        sampled_intensities = sampler.ddim_sample_loop(model=model,
                                                y=args.sample_settings,
                                                cfg_scale=args.cfg_scale,
                                                device=args.device,
                                                eta=1,       
                                                n=args.n_samples
                                                )
    elif s_type == 'edm':
        sampled_intensities = sampler.sample(
                                    resolution=args.resolution,
                                    device=args.device,
                                    settings=args.sample_settings,
                                    n_samples=args.n_samples,
                                    cfg_scale=args.cfg_scale,
                                    settings_dim=args.settings_dim
                                    )

    results_dir = os.path.join("results", args.run_name)
    
    save_images(
        sampled_intensities.cpu(),
        torch.Tensor(args.sample_intensities).cpu(),
        torch.Tensor(args.sample_settings).cpu(),
        torch.Tensor(wavelengths).cpu(),
        os.path.join(results_dir, f"{epoch}_ema"),
        epoch
    )

def plot_and_save(predicted, filename, title_suffix=""):
        """Helper function to plot predictions and save the image."""
        plt.figure(figsize=(12, 6), dpi=300)
        
        plt.plot(wavelengths, true_intensities, color='tab:orange', label="True spectrum", linewidth=2)
        
        if predicted.ndim == 1:  # Single prediction case
            plt.plot(wavelengths, predicted, color='tab:blue', alpha=1, label="Predicted sample 1")
        elif predicted.shape[0] == 2:  # Two predictions case
            plt.plot(wavelengths, predicted[0], color='tab:blue', alpha=0.7, label="Predicted sample 1")
            plt.plot(wavelengths, predicted[1], color='tab:green', alpha=0.7, label="Predicted sample 2")
        else:  # More than 2 predictions → plot as range
            min_intensities = np.min(predicted, axis=0)
            max_intensities = np.max(predicted, axis=0)
            plt.fill_between(wavelengths, min_intensities, max_intensities, color='tab:blue', alpha=0.3, label="Predicted range")

        plt.xlabel("Wavelengths")
        plt.ylabel("Intensity")
        plt.title(f"Epoch: {epoch} | Cond: [{settings_str}] {title_suffix}")
        plt.legend()
        plt.savefig(filename, dpi=300)
        plt.close()

def save_images(intensities, true_intensities, settings, wavelengths, base_path, epoch):
    """Saves predicted intensities as images with different visualization options."""
    
    settings_str = ', '.join([f"{val:.2f}" for val in settings[0].tolist()])
    intensities_np = np.array(intensities)[:, 0, :]
    n_samples = len(intensities_np)
    
    if n_samples == 1:
        plot_and_save(intensities_np[0], f"{base_path}_1vector.jpg")
    elif n_samples == 2:
        plot_and_save(intensities_np, f"{base_path}_2vectors.jpg")
        plot_and_save(intensities_np[0], f"{base_path}_1vector.jpg")
    else:
        plot_and_save(intensities_np[0], f"{base_path}_1vector.jpg")
        plot_and_save(intensities_np[:2], f"{base_path}_2vectors.jpg")
        plot_and_save(intensities_np, f"{base_path}_range.jpg")


def save_model(args, ema_model, optimizer, epoch=""):
    """Saves model checkpoints."""
    model_path = os.path.join("models", args.run_name)
    torch.save(ema_model.state_dict(), os.path.join(model_path, f"ema_ckpt{epoch}.pt"))
    torch.save(optimizer.state_dict(), os.path.join(model_path, f"optim{epoch}.pt"))

def evaluate_and_save_metrics(model, 
                              sampler, 
                              dataloader, 
                              device, 
                              epoch, 
                              s_type, 
                              args, 
                              file_name= "metrics_all_epochs.txt"):

    x_real, cond_vectors, predictions = predict(
        model,
        sampler,
        dataloader,
        device=device,
        n_samples=1,
        s_type=s_type,
        cfg_scale=args.cfg_scale,
        settings_dim=args.settings_dim
    )

    mse_error = np.mean((x_real - predictions) ** 2)
    print(f"MSE error: {mse_error:.6f}")

    wasserstein_distances, mean_wasserstein = calculate_wasserstein_distance(x_real, predictions)
    print(f"Mean Wasserstein Distance: {mean_wasserstein:.6f}")

    dtw_distances = calculate_dtw_distances(x_real, predictions)
    mean_dtw = np.mean(dtw_distances)
    print(f"Mean DTW Distance: {mean_dtw:.6f}")

    predictions_path = os.path.join("predictions", args.run_name, f"preds_epoch{epoch}.csv")
    save_predictions(x_real, cond_vectors, predictions, predictions_path)

    metrics_file_path = os.path.join("predictions", args.run_name, file_name)
    os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
    
    mode = 'a'
    if epoch == 1:
        mode = 'w'
    
    with open(metrics_file_path, mode) as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"MSE Error: {mse_error:.6f}\n")
        f.write(f"Mean Wasserstein Distance: {mean_wasserstein:.6f}\n")
        f.write(f"Mean DTW Distance: {mean_dtw:.6f}\n")
        f.write("-" * 50 + "\n")

def save_training_loss(epoch, loss_value, args):
    """Saves training loss after each epoch to a file."""
    loss_file_path = os.path.join("predictions", args.run_name, "training_loss.txt")
    os.makedirs(os.path.dirname(loss_file_path), exist_ok=True)

    mode = 'a'
    if epoch == 1:
        mode = 'w'

    with open(loss_file_path, mode) as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Training Loss: {loss_value:.6f}\n")
        f.write("-" * 50 + "\n")


