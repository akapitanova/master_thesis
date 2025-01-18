import copy

import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from .modules_edm import EDMPrecond
from src.diffusion import *
from src.utils import *
from src.ddim.modules_ddim import EMA
from .loss import EDMLoss

def setup_logging(run_name):
    os.makedirs("src/models", exist_ok=True)
    os.makedirs("src/results", exist_ok=True)
    os.makedirs(os.path.join("src/models", run_name), exist_ok=True)
    os.makedirs(os.path.join("src/results", run_name), exist_ok=True)

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
        model = EDMPrecond(img_resolution   = 1024,
                            img_channels    = 1,
                            label_dim       = 3,
                            use_fp16        = False,
                            sigma_min       = 0,
                            sigma_max       = float('inf'),
                            sigma_data      = 0.5,
                            model_type      = 'DhariwalUNet',
                            ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    #---------------------------------------------------------------------------

    sampler = EdmSampler(net=model, num_steps=100)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch)
    loss_fn = EDMLoss()

    logger = SummaryWriter(os.path.join("src/runs", args.run_name))

    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(device)

    model.train().requires_grad_(True)
    # Training loop
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        wavelengths = np.load('data/wavelengths.npy')

        for i, data in enumerate(pbar):
            vectors = data['data'].to(device)
            settings = data['settings'].to(device)

            if np.random.random() < 0.1:
                settings = None

            loss = loss_fn(net=model, images=vectors, labels=settings)

            # Accumulate gradients
            optimizer.zero_grad()
            loss.sum().backward()

            # Update weights
            optimizer.step()
            scheduler.step()

            # TODO
            ema.step_ema(ema_model, model)

            pbar.set_postfix({"_Loss": "{:.4f}".format(loss.sum())})
            logger.add_scalar("Loss", loss.sum(), global_step=epoch * l + i)

        # Save samples periodically
        if args.sample_freq and epoch % args.sample_freq == 0:# and epoch > 0:
            settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
            ema_sampled_vectors = sampler.sample(
                                            length=args.length,
                                            device=device,
                                            class_labels=settings,
                                            n_samples=args.n_samples
                                        )

            save_images(ema_sampled_vectors, os.path.join("results",
                                                          args.run_name,
                                                          wavelengths,
                                                          f"{epoch}_ema.jpg"))
            torch.save(ema_model.state_dict(), os.path.join("src/models",
                                                            args.run_name,
                                                            f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("src/models",
                                                            args.run_name,
                                                            f"optim.pt"))
    # Save final samples and checkpoints
    if not args.sample_freq:
        if args.sample_size:
            settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
            ema_sampled_vectors = sampler.sample(
                                            length=args.length,
                                            device=device,
                                            class_labels=settings,
                                            n_samples=args.n_samples
                                        )
            #save_samples(ema_sampled_vectors, os.path.join("../results",
             #                                              args.run_name))
    torch.save(ema_model.state_dict(), os.path.join("src/models",
                                                    args.run_name,
                                                    f"ema_ckpt.pt"))
    torch.save(optimizer.state_dict(), os.path.join("src/models",
                                                    args.run_name,
                                                    f"optim.pt"))

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "edm_e1000_bs16"
    args.epochs = 1000
    args.n_samples = 1
    #args.epochs = 1
    args.noise_steps = 1000
    args.beta_start = 1e-4
    args.beta_end = 0.02
    args.batch_size = 16
    # length of the input
    args.length = 1024
    #args.length = 512
    args.features = ['Stage3_OutputPower',
    'Stage3_Piezo',
    'stepper_diff']
    args.device = "cuda:0"
    #args.device = "cpu"
    args.lr = 1e-3
    args.grad_acc = 1
    args.sample_freq = 0
    data_path = "data/train_data_1024.csv"
    args.x_train, args.y_train = get_data(data_path)

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
