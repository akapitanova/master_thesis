import copy
import argparse

from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from modules1 import EMA, EDMPrecond, MaxScalePredictor
from diffusion1 import *
from train_utils import *
from dataset import *
from loss1 import EDMLoss
import torch.nn as nn
import numpy as np


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
        model = EDMPrecond(resolution   = args.length,
                           channels    = 1,
                           label_dim       = args.label_dim,
                           use_fp16        = False,
                           sigma_min       = 0,
                           sigma_max       = float('inf'),
                           sigma_data      = 0.5,
                           model_type      = 'UNet_conditional',
                           device          = device,
                           dropout_rate    = args.dropout_rate
                           ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    #---------------------------------------------------------------------------
    scale_predictor = MaxScalePredictor(input_dim=args.label_dim).to(device)
    scale_optimizer = optim.AdamW(scale_predictor.parameters(), lr=args.lr)

    sampler = EdmSampler(net=model, num_steps=args.edm_num_steps)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch)
    scale_scheduler = CosineAnnealingLR(scale_optimizer, T_max=args.epochs * steps_per_epoch)

    loss_fn = EDMLoss()
    scale_loss_fn = nn.MSELoss()

    logger = SummaryWriter(os.path.join("runs", args.run_name))

    ema = EMA(args.ema_decay)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(device)

    model.train().requires_grad_(True)
    scale_predictor.train()
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        wavelengths = np.load('../data/wavelengths.npy')

        for i, data in enumerate(pbar):
            vectors = data['data'].to(device)
            settings = data['settings'].to(device)

            # Compute true max scale
            true_max_scale = vectors.max(dim=-1, keepdim=True)[0].squeeze(-1)

            # Normalize the spectrum
            #normalized_vectors = vectors / (abs(true_max_scale) + 1e-8)
            #normalized_vectors = vectors
            lower_bound = vectors.min(dim=-1, keepdim=True)[0]
            upper_bound = vectors.max(dim=-1, keepdim=True)[0]
            norm_min = torch.tensor([-1]).reshape(-1, 1).to(device)
            norm_max = torch.tensor([1]).reshape(-1, 1).to(device)
            normalized_vectors = norm_min + (vectors - lower_bound) * (norm_max - norm_min) / (upper_bound - lower_bound)
            #print(f'vectors: {vectors.shape}')
            #print(f'normalized_vectors: {normalized_vectors.shape}')
            

            # **Train max scale predictor based on the conditional vector**
            predicted_scale = scale_predictor(settings)
            #print(f'true_max_scale: {true_max_scale}')
            #print(f'predicted_scale: {predicted_scale}')
            scale_loss = scale_loss_fn(predicted_scale, true_max_scale)

            scale_optimizer.zero_grad()
            scale_loss.backward()
            scale_optimizer.step()

            scale_scheduler.step()

            # classifier-free guidance
            if np.random.random() < args.cfg_scale_train:
                settings = None
                
            # training step
            loss = loss_fn(net=model, images=normalized_vectors, labels=settings)
            #print(type(loss))
            
            if torch.isnan(loss.mean()):
                print(f'lower_bound: {lower_bound}')
                print(f'upper_bound: {upper_bound}')
                print(f'loss: {loss}')
                break


            optimizer.zero_grad()
            loss.mean().backward()

            optimizer.step()
            scheduler.step()

            ema.step_ema(ema_model, model)
            
            #pbar.set_postfix({"_Loss": "{:.4f}".format(loss.mean())})
            pbar.set_postfix({
                                "Loss": "{:.4f}".format(loss.mean()),
                                "Scale_Loss": "{:.4f}".format(scale_loss.item())
                            })
            logger.add_scalar("Loss", loss.mean(), global_step=epoch * l + i)
            logger.add_scalar("Scale Loss", scale_loss.item(), global_step=epoch * l + i)
            


        # Save samples periodically
        if args.sample_freq and epoch % args.sample_freq == 0:
            settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
            #predicted_max_scale = scale_predictor(settings).detach().cpu().numpy()
            predicted_max_scale = scale_predictor(settings).detach()
            #print(predicted_max_scale)

            ema_sampled_vectors = sampler.sample(
                                            length=args.length,
                                            device=device,
                                            class_labels=settings,
                                            n_samples=args.n_samples,
                                            cfg_scale=args.cfg_scale,
                                            label_dim = args.label_dim
                                        )
            print(ema_sampled_vectors)
            print(predicted_max_scale.reshape(-1, 1, 1))
            lower_bound = ema_sampled_vectors.min(dim=-1, keepdim=True)[0]
            upper_bound = ema_sampled_vectors.max(dim=-1, keepdim=True)[0]
            norm_min = torch.tensor([-1]).reshape(-1, 1, 1).to(device)
            norm_max = predicted_max_scale.reshape(-1, 1, 1)
            ema_sampled_vectors = norm_min + (ema_sampled_vectors - lower_bound) * (norm_max - norm_min) / (upper_bound - lower_bound)
            print(ema_sampled_vectors)

            save_images(torch.Tensor(ema_sampled_vectors[0, :, :]).to('cpu'),
                        torch.Tensor(args.sample_spectrum_real).to('cpu'),
                        torch.Tensor(args.sample_settings).to('cpu'),
                        torch.Tensor(wavelengths).to('cpu'),
                        os.path.join("results",
                                      args.run_name,
                                      f"{epoch}_ema.jpg"),
                        epoch)
            torch.save(ema_model.state_dict(), os.path.join("models",
                                                            args.run_name,
                                                            f"ema_ckpt_epoch{epoch}.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models",
                                                            args.run_name,
                                                            f"optim_epoch{epoch}.pt"))
            torch.save(scale_predictor.state_dict(), os.path.join("models", 
                                                                  args.run_name,
                                                                  f"scale_predictor_epoch{epoch}.pt"))
        # Save final samples and checkpoints
        torch.save(ema_model.state_dict(), os.path.join("models",
                                                        args.run_name,
                                                        f"ema_ckpt.pt"))
        torch.save(optimizer.state_dict(), os.path.join("models",
                                                        args.run_name,
                                                        f"optim.pt"))
        torch.save(scale_predictor.state_dict(), os.path.join("models", 
                                                              args.run_name,
                                                              f"scale_predictor.pt"))

    settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
    predicted_max_scale = scale_predictor(settings).detach()
    print(predicted_max_scale)
    ema_sampled_vectors = sampler.sample(
                                    length=args.length,
                                    device=device,
                                    class_labels=settings,
                                    n_samples=args.n_samples,
                                    cfg_scale=args.cfg_scale,
                                    label_dim = args.label_dim
                                )
    print(ema_sampled_vectors)
    lower_bound = ema_sampled_vectors.min(dim=-1, keepdim=True)[0]
    upper_bound = ema_sampled_vectors.max(dim=-1, keepdim=True)[0]
    norm_min = torch.tensor([-1]).reshape(-1, 1, 1).to(device)
    norm_max = predicted_max_scale.reshape(-1, 1, 1)
    ema_sampled_vectors = norm_min + (ema_sampled_vectors - lower_bound) * (norm_max - norm_min) / (upper_bound - lower_bound) 
    print(ema_sampled_vectors)

    save_images(torch.Tensor(ema_sampled_vectors[0, :, :]).to('cpu'),
                torch.Tensor(args.sample_spectrum_real).to('cpu'),
                torch.Tensor(args.sample_settings).to('cpu'),
                torch.Tensor(wavelengths).to('cpu'),
                os.path.join("results",
                             args.run_name,
                             f"{epoch}_final_ema.jpg"),
                epoch)


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
    parser.add_argument("--length", 
                        type=int, 
                        default=1024, 
                        help="Input length")
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
    parser.add_argument("--edm_num_steps", 
                        type=int, 
                        default=100, 
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
                        default="../data/train_data_1024_[-1,1].csv", 
                        help="Path to training data")
    parser.add_argument("--sample_spectrum_path", 
                        type=str, 
                        default="../data/sample_spectrum.csv", 
                        help="Path to sample spectrum")
    parser.add_argument("--label_dim", 
                        type=int, 
                        default=3, 
                        help="Number of labels (conditioning vector size)")
    
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
