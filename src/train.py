import os
import logging
import copy

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision
import torchvision.transforms.functional as f

from tqdm import tqdm
from torchsummary import summary

from modules import UNet_conditional, EMA
from diffusion import *
from utils import *

def save_samples(images, folder="samples", start_index=0):
    # print(images.shape)
    ndarr = images.to('cpu').numpy()
    indexes = range(start_index, start_index + len(ndarr))
    for i, im in zip(indexes, ndarr):
        cv2.imwrite(folder + "/" + str(i) + ".png", im)

# TODO
def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

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

    #for i, data in enumerate(train_dataloader):
    #    if i == 282:
    #        print(data['data'])

    gradient_acc = args.grad_acc
    l = len(train_dataloader)
    steps_per_epoch = l / gradient_acc

    #---------------------------------------------------------------------------
    if not model:
        print("Training from scratch")
        model = UNet_conditional(length=args.length,
                                 device=args.device,
                                 feat_num=len(args.features)
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
                              section_counts=[40],
                              noise_steps=args.noise_steps,
                              length=args.length,
                              device=device)

    logger = SummaryWriter(os.path.join("runs", args.run_name))

    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(device)

    # Training loop
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs - 1}")

        for i, data in enumerate(pbar):
            vectors = data['data'].to(device)
            settings = data['settings'].to(device)

            t = diffusion.sample_timesteps(vectors.shape[0], all_same=False).to(device)
            x_t, noise = diffusion.noise_images(vectors, t)

            if np.random.random() < 0.1:
                settings = None

            predicted_noise = model(x_t, t, settings)


            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)
            scheduler.step()

            pbar.set_postfix({"_MSE": "{:.4f}".format(loss.item())})

            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

            break

        if args.sample_freq and epoch % args.sample_freq == 0:# and epoch > 0:
            settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
            ema_sampled_vectors = sampler.ddim_sample_loop(model=ema_model,
                                                          y=settings,
                                                          cfg_scale=3,
                                                          device=device,
                                                          eta=1,
                                                          n=args.batch_size,
                                                          #resize=args.real_size
                                                           )
            # TODO
            #save_images(ema_sampled_vectors, os.path.join("results",
            #                                              args.run_name,
            #                                              f"{epoch}_ema.jpg"))
            torch.save(ema_model.state_dict(), os.path.join("models",
                                                            args.run_name,
                                                            f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models",
                                                            args.run_name,
                                                            f"optim.pt"))

    if not args.sample_freq:
        if args.sample_size:
            settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
            ema_sampled_vectors = sampler.ddim_sample_loop(model=ema_model,
                                                          y=settings,
                                                          cfg_scale=3,
                                                          device=device,
                                                          eta=1,
                                                          n=args.batch_size,
                                                          #resize=args.real_size
                                                           )
            save_samples(ema_sampled_vectors, os.path.join("results",
                                                           args.run_name))
        torch.save(ema_model.state_dict(), os.path.join("models",
                                                        args.run_name,
                                                        f"ema_ckpt.pt"))
        torch.save(optimizer.state_dict(), os.path.join("models",
                                                        args.run_name,
                                                        f"optim.pt"))

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "test"
    args.epochs = 601
    #args.epochs = 1
    args.noise_steps = 1000
    args.beta_start = 1e-4
    args.beta_end = 0.02
    args.batch_size = 4
    # length of the input
    args.length = 1024
    args.features = ['Stage3_OutputPower',
    'Stage3_Piezo',
    'stepper_diff']
    args.device = "cuda:0"
    #args.device = "cpu"
    args.lr = 1e-3
    args.grad_acc = 1
    args.sample_freq = 0
    data_path = "../data/train_data.csv"
    args.x_train, args.y_train = get_data(data_path)
    #args.sample_settings = [32.,15.,15.]
    # cond vector pro zkusebni datapoint behem prubezneho ukladani v trenovani
    args.sample_settings = [1.0, 0.604, 0.0]
    args.sample_size = 8

    #model = UNet_conditional(length=1024,
    #                         feat_num=3,
    #                         device=args.device).to(args.device)
    #ckpt = torch.load("models/transfered.pt", map_location=args.device)
    #model.load_state_dict(ckpt)
    train(args, model=None)

if __name__ == '__main__':
    launch()