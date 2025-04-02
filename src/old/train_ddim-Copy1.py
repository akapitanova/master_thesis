import copy
import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from modules import EMA, UNet_conditional
from diffusion import SpacedDiffusion
from dataset import CustomDataset, get_data
from train_utils import setup_logging, normalize_vectors, save_samples, save_images, save_model

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def train(args, model=None, finetune=False):
    setup_logging(args.run_name)
    device = args.device
    x_train = args.x_train
    y_train = args.y_train

    train_dataset = CustomDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    gradient_acc = args.grad_acc
    l = len(train_dataloader)
    steps_per_epoch = l / gradient_acc

    #---------------------------------------------------------------------------
    if not model:
        print("Training from scratch")
        model = UNet_conditional(length=args.length,
                                 device=args.device,
                                 feat_num=args.label_dim,
                                 dropout_rate=args.dropout_rate,
                                 sampler_type="DDIM",
                                 ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    #---------------------------------------------------------------------------
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs*steps_per_epoch)
    mse = nn.MSELoss()
    betas = prepare_noise_schedule(args.noise_steps,
                                   args.beta_start,
                                   args.beta_end)

    diffusion = GaussianDiffusion(betas=betas,
                                  noise_steps=args.noise_steps,
                                  length=args.length,
                                  device=device)
    sampler = SpacedDiffusion(beta_start=args.beta_start,
                              beta_end=args.beta_end,
                              section_counts=[args.section_counts],
                              noise_steps=args.noise_steps,
                              length=args.length,
                              device=device)

    logger = SummaryWriter(os.path.join("runs", args.run_name))

    ema = EMA(args.ema_decay)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(device)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        wavelengths = np.load('../data/wavelengths.npy')
        
        for i, data in enumerate(pbar):
            vectors = data['data'].to(device)
            settings = data['settings'].to(device)

            t = diffusion.sample_timesteps(vectors.shape[0], all_same=False).to(device)
            x_t, noise = diffusion.noise_images(vectors, t)

            # classifier-free guidance
            if np.random.random() < args.cfg_scale_train:
                settings = None

            # training step
            predicted_noise = model(x_t, t, settings)

            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            ema.step_ema(ema_model, model)

            pbar.set_postfix({"_MSE": "{:.4f}".format(loss.item())})
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        # Save samples periodically
        if args.sample_freq and epoch % args.sample_freq == 0: 
            settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
            ema_sampled_vectors = sampler.ddim_sample_loop(
                                                        model=ema_model,
                                                        y=settings,
                                                        cfg_scale=args.cfg_scale,
                                                        device=device,
                                                        eta=1,
                                                        n=args.n_samples,
                                                        )

            #print(torch.Tensor(ema_sampled_vectors).shape)

            save_images(torch.Tensor(ema_sampled_vectors).unsqueeze(0).to('cpu'),
                        torch.Tensor(args.sample_spectrum_real).to('cpu'),
                        torch.Tensor(args.sample_settings).to('cpu'),
                        torch.Tensor(wavelengths).to('cpu'),
                        os.path.join("results",
                                    args.run_name,
                                    f"{epoch}_ema.jpg"),
                        epoch)
            torch.save(ema_model.state_dict(), os.path.join("models",
                                                            args.run_name,
                                                            f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models",
                                                            args.run_name,
                                                            f"optim.pt"))

        # Save final samples and checkpoints
        torch.save(ema_model.state_dict(), os.path.join("models",
                                                        args.run_name,
                                                        f"ema_ckpt.pt"))
        torch.save(optimizer.state_dict(), os.path.join("models",
                                                        args.run_name,
                                                        f"optim.pt"))

    settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
    ema_sampled_vectors = sampler.ddim_sample_loop(model=ema_model,
                                                   y=settings,
                                                   cfg_scale=args.cfg_scale,
                                                   device=device,
                                                   eta=1,
                                                   n=args.n_samples,
                                                   )

    save_images(torch.Tensor(ema_sampled_vectors).unsqueeze(0).to('cpu'),
                        torch.Tensor(args.sample_spectrum_real).to('cpu'),
                        torch.Tensor(args.sample_settings).to('cpu'),
                        torch.Tensor(wavelengths).to('cpu'),
                        os.path.join("results",
                        args.run_name,
                        f"{epoch}_final_ema.jpg"),
                        epoch)

def launch():
    parser = argparse.ArgumentParser(description="Train DDIM Model")
    
    # Model & Training Parameters
    parser.add_argument("--run_name", 
                        type=str, 
                        default="ddim_e300_bs16", 
                        help="Run name for logging")
    parser.add_argument("--epochs", 
                        type=int, 
                        default=300, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=16, 
                        help="Batch size")
    parser.add_argument("--length", 
                        type=int, 
                        default=1024, 
                        help="Input length")
    parser.add_argument("--device", 
                        type=str, 
                        default="cuda:1", 
                        help="Training device")
    parser.add_argument("--lr", 
                        type=float, 
                        default=1e-3, 
                        help="Learning rate")
    parser.add_argument("--grad_acc", 
                        type=int, 
                        default=1, 
                        help="Gradient accumulation steps")
    parser.add_argument("--sample_freq", 
                        type=int, 
                        default=10, 
                        help="Frequency of saving samples")
    parser.add_argument("--n_samples", 
                        type=int, 
                        default=1, 
                        help="Number of samples to generate")
    parser.add_argument("--dropout_rate", 
                        type=float, 
                        default=0.05, 
                        help="Dropout rate")
    parser.add_argument("--ema_decay", 
                        type=float, 
                        default=0.995, 
                        help="EMA decay rate")
    parser.add_argument("--noise_steps", 
                        type=int, 
                        default=1000, 
                        help="Number of noise steps")
    parser.add_argument("--cfg_scale_train", 
                        type=float, 
                        default=0.0, 
                        help="Classifier-Free Guidance scale training (0 - no CFG, 1 - without settings)")
    parser.add_argument("--cfg_scale", 
                        type=float, 
                        default=3, 
                        help="Classifier-Free Guidance scale training (default 3)")
    parser.add_argument("--beta_start", 
                        type=float, 
                        default=1e-4, 
                        help="Beta start value")
    parser.add_argument("--beta_end", 
                        type=float, 
                        default=0.02, 
                        help="Beta end value")
    parser.add_argument("--data_path", 
                        type=str, 
                        default="../data/train_data_stg7_norm.csv", 
                        help="Path to training data")
    parser.add_argument("--sample_spectrum_path", 
                        type=str, 
                        default="../data/sample_spectrum_stgF.csv", 
                        help="Path to sample spectrum")
    parser.add_argument("--label_dim", 
                        type=int, 
                        default=3, 
                        help="Number of labels (conditioning vector size)")
    parser.add_argument("--section_counts", 
                        type=int, 
                        default=40, 
                        help="Section counts for SpacedDiffusion")

    args = parser.parse_args()

    # Load dataset
    data_path = args.data_path
    args.x_train, args.y_train = get_data(data_path)

    # Load sample spectrum
    sample_spectrum_path = args.sample_spectrum_path
    data = pd.read_csv(sample_spectrum_path)
    args.sample_spectrum_real = np.array(data['intensities'].apply(lambda x: eval(x) if isinstance(x, str) else x).iloc[0])
    args.sample_settings = np.array(data['cond_vector'].apply(lambda x: eval(x) if isinstance(x, str) else x).iloc[0])

    # Train model
    train(args, model=None)

if __name__ == '__main__':
    launch()
