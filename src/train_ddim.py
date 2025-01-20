import copy

import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from modules import UNet_conditional, EMA
from diffusion import *
from utils import *
from dataset import *

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
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        wavelengths = np.load('../data/wavelengths.npy')
        for i, data in enumerate(pbar):
            vectors = data['data'].to(device)
            settings = data['settings'].to(device)

            t = diffusion.sample_timesteps(vectors.shape[0], all_same=False).to(device)
            x_t, noise = diffusion.noise_images(vectors, t)


            if np.random.random() < 0.1:
                settings = None

            #x_t = torch.unsqueeze(x_t, 1)
            predicted_noise = model(x_t, t, settings)


            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)
            scheduler.step()

            pbar.set_postfix({"_MSE": "{:.4f}".format(loss.item())})

            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        # Save samples periodically
        if args.sample_freq and epoch % args.sample_freq == 0:  # and epoch > 0:
            settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
            ema_sampled_vectors = sampler.ddim_sample_loop(model=ema_model,
                                                      y=settings,
                                                      cfg_scale=3,
                                                      device=device,
                                                      eta=1,
                                                      n=args.n_samples,
                                                      #resize=args.real_size
                                                       )

            save_images(torch.Tensor(ema_sampled_vectors).to('cpu'),
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

        torch.save(ema_model.state_dict(), os.path.join("models",
                                                        args.run_name,
                                                        f"ema_ckpt.pt"))
        torch.save(optimizer.state_dict(), os.path.join("models",
                                                        args.run_name,
                                                        f"optim.pt"))

    settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
    ema_sampled_vectors = sampler.ddim_sample_loop(model=ema_model,
                                                   y=settings,
                                                   cfg_scale=3,
                                                   device=device,
                                                   eta=1,
                                                   n=args.n_samples,
                                                   # resize=args.real_size
                                                   )

    save_images(torch.Tensor(ema_sampled_vectors).to('cpu'),
                        torch.Tensor(args.sample_spectrum_real).to('cpu'),
                        torch.Tensor(args.sample_settings).to('cpu'),
                        torch.Tensor(wavelengths).to('cpu'),
                os.path.join("results",
                             args.run_name,
                             f"{epoch}_final_ema.jpg"),
                epoch)

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "ddim_e300_bs16"
    args.epochs = 300
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
    args.device = "cuda:1"
    #args.device = "cpu"
    args.lr = 1e-3
    args.grad_acc = 1
    args.sample_freq = 10
    args.n_samples = 2
    data_path = "../data/train_data_1024_[-1,1].csv"
    args.x_train, args.y_train = get_data(data_path)

    # cond vector pro zkusebni datapoint behem prubezneho ukladani v trenovani
    sample_spectrum_path = '../data/sample_spectrum.csv'
    data = pd.read_csv(sample_spectrum_path)
    args.sample_spectrum_real = np.array(data['intensities'].apply(lambda x: eval(x) if isinstance(x, str) else x).iloc[0])
    args.sample_settings = np.array(data['cond_vector'].apply(lambda x: eval(x) if isinstance(x, str) else x).iloc[0])
    train(args, model=None)

if __name__ == '__main__':
    launch()
