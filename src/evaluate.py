import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from modules import UNet_conditional
from diffusion import *
from utils import *


def predict(model,
            sampler,
            test_dl,
            device,
            n_samples=1):
    """
    Return predictions
    """
    x_real = []
    cond_vectors = []
    predictions = []

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dl, desc="Testing loop")):
            vectors = data['data'].to(device)
            settings = data['settings'].to(device)

            pred = sampler.ddim_sample_loop(model=model,
                                            y=settings,
                                            cfg_scale=1,
                                            device=device,
                                            eta=1,
                                            n=n_samples
                                            )

            x_real.extend(vectors.cpu().tolist() * n_samples)
            cond_vectors.extend(settings.cpu().tolist() * n_samples)
            predictions.append(pred.cpu().tolist())

    return x_real, cond_vectors, predictions


def evaluate(model,
             sampler,
             device,
             test_csv_path,
             n_samples=1):
    """
    Evaluate predictions
    """
    # Load the test dataset
    x_test, y_test = get_data(test_csv_path)

    test_dataset = CustomDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False)

    x_real, cond_vectors, predictions = predict(model,
                                  sampler,
                                  test_dataloader,
                                  device=device,
                                  n_samples=n_samples)

    # intesities are normalized
    #x_real = [[x * 3925 for x in row] for row in x_real]

    return x_real, cond_vectors, predictions

