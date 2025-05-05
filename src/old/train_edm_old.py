import copy
import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from modules import EMA, EDMPrecond, MaxScalePredictor
from diffusion import EdmSampler
from train_utils import setup_logging, normalize_vectors, save_samples, save_images, save_model, evaluate_and_save_metrics, save_training_loss
from dataset import CustomDataset, get_data
from loss import EDMLoss
from evaluate import normalize_sampled_vectors


def train(args, model=None, scale_predictor=None, finetune=False):
    setup_logging(args.run_name)
    device = args.device
    
    # Load dataset
    x_train, y_train = args.intensities, args.settings
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    train_dataset = CustomDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Load the val dataset
    x_val, y_val = get_data(args.val_data_path)

    val_dataset = CustomDataset(x_val, y_val)
    val_dataloader = DataLoader(val_dataset,
                                 batch_size=1,
                                 shuffle=False)

    gradient_acc = args.grad_acc
    l = len(train_dataloader)
    steps_per_epoch = l / gradient_acc

    # Initialize Model & Optimizers
    if not model:
        print("Training from scratch")
        model = EDMPrecond(resolution      = args.resolution,
                           settings_dim    = args.settings_dim,
                           device          = device,
                           dropout_rate    = args.dropout_rate
                           ).to(device)
        scale_predictor = MaxScalePredictor(input_dim=args.settings_dim).to(device)

    #---------------------------------------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scale_optimizer = optim.AdamW(scale_predictor.parameters(), lr=0.01)

    sampler = EdmSampler(net=model, num_steps=args.noise_steps)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch, eta_min = 0)
    scale_scheduler = CosineAnnealingLR(scale_optimizer, T_max=args.epochs * steps_per_epoch, eta_min = 0)

    loss_fn = EDMLoss()
    scale_loss_fn = nn.MSELoss()

    logger = SummaryWriter(os.path.join("runs", args.run_name))

    ema = EMA(args.ema_decay)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(device)

    model.train().requires_grad_(True)
    scale_predictor.train()
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        wavelengths = np.load('../data/wavelengths.npy')

        for i, data in enumerate(pbar):
            intensities = data['data'].to(device)
            settings = data['settings'].to(device)

            # Compute true max scale
            true_max_scale = intensities.max(dim=-1, keepdim=True)[0].squeeze(-1)

            # Normalize the spectrum
            normalized_intensities = normalize_vectors(intensities, device)
            
            # **Train max scale predictor based on the conditional vector**
            predicted_scale = scale_predictor(settings)
            scale_loss = scale_loss_fn(predicted_scale, true_max_scale)

            scale_optimizer.zero_grad()
            scale_loss.backward()
            scale_optimizer.step()

            scale_scheduler.step()

            # classifier-free guidance
            if np.random.random() < args.cfg_scale_train:
                settings = None
                
            # training step
            loss = loss_fn(net=model, 
                           y=normalized_intensities, 
                           settings=settings)

            optimizer.zero_grad()
            loss.mean().backward()

            optimizer.step()
            scheduler.step()

            ema.step_ema(ema_model, model)
            
            pbar.set_postfix({
                                "Loss": "{:.4f}".format(loss.mean()),
                                "Scale_Loss": "{:.4f}".format(scale_loss.item())
                            })
            logger.add_scalar("Loss", loss.mean(), global_step=epoch * l + i)
            logger.add_scalar("Scale Loss", scale_loss.item(), global_step=epoch * l + i)

        # Save samples periodically
        if args.sample_freq and epoch % args.sample_freq == 0:
            save_samples(epoch, scale_predictor, sampler, wavelengths, args)

        # Save model after each epoch
        save_model(args, ema_model, optimizer, scale_predictor, epoch)

        # evaluate on val data
        if args.evaluation: 
            if args.epochs <= 10:
                evaluate_and_save_metrics(ema_model, scale_predictor, sampler, val_dataloader, device, epoch, "edm", args)
            elif epoch == 1:
                evaluate_and_save_metrics(ema_model, scale_predictor, sampler, val_dataloader, device, epoch, "edm", args)
            elif args.epochs <= 20 and epoch % 2 == 0:
                evaluate_and_save_metrics(ema_model, scale_predictor, sampler, val_dataloader, device, epoch, "edm", args)
            elif epoch % 5 == 0:
                evaluate_and_save_metrics(ema_model, scale_predictor, sampler, val_dataloader, device, epoch, "edm", args)

        # Save training loss after each epoch
        save_training_loss(epoch, loss.mean(), scale_loss.item(), args)

    # Save final samples and model
    save_samples(epoch, scale_predictor, sampler, wavelengths, args)
    save_model(args, ema_model, optimizer, scale_predictor)


def launch():    
    parser = argparse.ArgumentParser(description="Train EDM Model")
    
    # Model & Training Parameters
    parser.add_argument("--run_name", 
                        type=str, 
                        default="edm_e300_bs16_do5_no-cg", 
                        help="Run name for logging")
    parser.add_argument("--epochs", 
                        type=int, 
                        default=300, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=16, 
                        help="Batch size")
    parser.add_argument("--resolution", 
                        type=int, 
                        default=1024, 
                        help="Input resolution")
    parser.add_argument("--device", 
                        type=str, 
                        default="cuda:1", 
                        help="Training device (cuda or cpu)")
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
                        default=1, 
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
                        default=20, 
                        help="Number of steps in EdmSampler")
    parser.add_argument("--cfg_scale_train", 
                        type=float, 
                        default=0.0, 
                        help="Classifier-Free Guidance scale training (0 - no CFG, 1 - without settings)")
    parser.add_argument("--cfg_scale", 
                        type=float, 
                        default=3, 
                        help="Classifier-Free Guidance scale training (default 3)")
    parser.add_argument("--data_path", 
                        type=str, 
                        default="../data/train_data_stg7_norm.csv", 
                        help="Path to training data")
    parser.add_argument("--val_data_path", 
                        type=str, 
                        default="../data/val_data_stg7_clipped.csv", 
                        help="Path to validation data")
    parser.add_argument("--sample_spectrum_path", 
                        type=str, 
                        default="../data/sample_spectrum_stgF.csv", 
                        help="Path to sample spectrum")
    parser.add_argument("--settings_dim", 
                        type=int, 
                        default=3, 
                        help="Number of settings (conditioning vector size)")
    parser.add_argument("--model_path", 
                        type=str, 
                        default=None, 
                        help="Path to pretrained model")
    parser.add_argument("--scale_predictor_path", 
                        type=str, 
                        default=None, 
                        help="Path to pretrained scale model")
    
    parser.add_argument("--evaluation",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Enable evaluation (default: True)")
    
    args = parser.parse_args()

    # Load dataset
    data_path = args.data_path
    args.intensities, args.settings = get_data(data_path)

    # Load sample spectrum
    sample_spectrum_path = args.sample_spectrum_path
    data = pd.read_csv(sample_spectrum_path)
    args.sample_intensities = np.array(data['intensities'].apply(lambda x: eval(x) if isinstance(x, str) else x).iloc[0])
    args.sample_settings = torch.Tensor(data['cond_vector'].apply(lambda x: eval(x) if isinstance(x, str) else x).iloc[0]).to(args.device).unsqueeze(0)

    # Load the models if given
    model = None
    scale_predictor = None
    
    if args.model_path and args.scale_predictor_path:
        model = EDMPrecond(device=device).to(device)
        ckpt = torch.load(model_path,
                          map_location=device,
                          weights_only=True)
        model.load_state_dict(ckpt)

        scale_predictor = MaxScalePredictor().to(device)
        ckpt_sp = torch.load(scale_predictor_path,
                             map_location=device,
                             weights_only=True
                            )
        scale_predictor.load_state_dict(ckpt_sp)
    
    # Train model
    train(args, model=model, scale_predictor=scale_predictor)

if __name__ == '__main__':
    launch()
