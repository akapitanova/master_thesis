import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse

from modules import UNet_conditional, EDMPrecond
from diffusion import EdmSampler, SpacedDiffusion
from dataset import CustomDataset, get_data


def sample_from_model(model, sampler, cond_dataloader, device, n_samples=1, s_type='edm', cfg_scale=1, settings_dim=13):
    cond_vectors = np.empty((0,))
    predictions = np.empty((0,))

    model.eval()
    with torch.no_grad():
        for data in tqdm(cond_dataloader, desc="Sampling"):
            settings = data['settings'].to(device)
            resolution = data.get('resolution', None)
            if resolution is not None:
                resolution = resolution.to(device)

            if s_type == 'ddim':
                pred = sampler.ddim_sample_loop(
                    model=model,
                    y=settings,
                    cfg_scale=cfg_scale,
                    device=device,
                    eta=1,
                    n=n_samples
                )
            elif s_type == 'edm':
                pred = sampler.sample(
                    resolution=resolution.item() if resolution is not None else 100,  # default/fallback
                    device=device,
                    settings=settings,
                    n_samples=n_samples,
                    cfg_scale=cfg_scale,
                    settings_dim=settings_dim
                )

            settings_np = settings.cpu().numpy().repeat(n_samples, axis=0)
            pred_np = pred.cpu().numpy()

            cond_vectors = np.concatenate((cond_vectors, settings_np), axis=0) if cond_vectors.size else settings_np
            predictions = np.concatenate((predictions, pred_np), axis=0) if predictions.size else pred_np

    return cond_vectors, predictions[:, 0, :]


def save_samples(cond_vectors, predictions, output_path):
    cond_str = [','.join(map(str, row)) for row in cond_vectors]
    pred_str = [','.join(map(str, row)) for row in predictions]
    
    df = pd.DataFrame({'cond_vectors': cond_str, 'samples': pred_str})
    df.to_csv(output_path, index=False)
    print(f"Samples saved to {output_path}")
    return df
