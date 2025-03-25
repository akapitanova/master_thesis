import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from evaluate import normalize_sampled_vectors


def setup_logging(run_name):
    """Creates necessary directories for model training and results storage."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def normalize_vectors(vectors, device):
    """Normalizes input vectors between -1 and 1."""
    lower_bound = vectors.min(dim=-1, keepdim=True)[0]
    upper_bound = vectors.max(dim=-1, keepdim=True)[0]
    norm_min = torch.tensor([-1], device=device).reshape(-1, 1)
    norm_max = torch.tensor([1], device=device).reshape(-1, 1)
    return norm_min + (vectors - lower_bound) * (norm_max - norm_min) / (upper_bound - lower_bound)

def save_samples(epoch, 
                 scale_predictor, 
                 sampler, 
                 wavelengths,
                 args,
                 s_type='edm',
                 model=None
                ):
    """Generates and saves samples using the trained model."""
    #settings = torch.Tensor(settings).to(device).unsqueeze(0)
    predicted_max_scale = scale_predictor(args.sample_settings).detach()

    
    if s_type == 'ddim':
        sampled_intensities = sampler.ddim_sample_loop(model=model,
                                                y=args.sample_settings,
                                                cfg_scale=args.cfg_scale,
                                                device=args.device,
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

    sampled_intensities = normalize_sampled_vectors(sampled_intensities, args.device, predicted_max_scale)

    save_images(
        sampled_intensities[0].cpu(),
        torch.Tensor(args.sample_intensities).cpu(),
        torch.Tensor(args.sample_settings).cpu(),
        torch.Tensor(wavelengths).cpu(),
        os.path.join("results", args.run_name, f"{epoch}_ema.jpg"),
        epoch
    )

def save_images(intensities,
                true_intensities,
                settings,
                wavelengths,
                path,
                epoch):
    plt.figure(figsize=(12, 6))
    settings_str = ', '.join([f"{val:.2f}" for val in settings[0].tolist()])
    for i, vector in enumerate(intensities):
        plt.plot(wavelengths, vector, label=f"Predicted sample")
    plt.plot(wavelengths, true_intensities, label=f"True spectrum")
    plt.xlabel("Wavelengths")
    plt.ylabel("Intensity")
    plt.title(f"Epoch: {epoch} | Cond: [{settings_str}]")
    plt.legend()
    plt.savefig(path)
    plt.close()

def save_model(args, ema_model, optimizer, scale_predictor, epoch=""):
    """Saves model checkpoints."""
    model_path = os.path.join("models", args.run_name)
    torch.save(ema_model.state_dict(), os.path.join(model_path, f"ema_ckpt{epoch}.pt"))
    torch.save(optimizer.state_dict(), os.path.join(model_path, f"optim{epoch}.pt"))
    torch.save(scale_predictor.state_dict(), os.path.join(model_path, f"scale_predictor{epoch}.pt"))
