import copy
import argparse

from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from modules import EMA, EDMPrecond
from diffusion import *
from utils import *
from dataset import *
from loss import EDMLoss

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

    sampler = EdmSampler(net=model, num_steps=args.edm_num_steps)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch)
    loss_fn = EDMLoss()

    logger = SummaryWriter(os.path.join("runs", args.run_name))

    ema = EMA(args.ema_decay)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(device)

    model.train().requires_grad_(True)
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        wavelengths = np.load('../data/wavelengths.npy')

        for i, data in enumerate(pbar):
            vectors = data['data'].to(device)
            settings = data['settings'].to(device)

            # classifier-free guidance
            if np.random.random() < args.cfg_scale:
                settings = None
                
            # training step
            loss = loss_fn(net=model, images=vectors, labels=settings)

            optimizer.zero_grad()
            loss.mean().backward()

            optimizer.step()
            scheduler.step()

            ema.step_ema(ema_model, model)

            pbar.set_postfix({"_Loss": "{:.4f}".format(loss.mean())})
            logger.add_scalar("Loss", loss.mean(), global_step=epoch * l + i)

        # Save samples periodically
        if args.sample_freq and epoch % args.sample_freq == 0:
            settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
            ema_sampled_vectors = sampler.sample(
                                            length=args.length,
                                            device=device,
                                            class_labels=settings,
                                            n_samples=args.n_samples
                                        )

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
    ema_sampled_vectors = sampler.sample(
                                    length=args.length,
                                    device=device,
                                    class_labels=settings,
                                    n_samples=args.n_samples
                                )

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
    parser.add_argument("--cfg_scale", 
                        type=float, 
                        default=0.0, 
                        help="Classifier-Free Guidance scale (0 - no CFG)")
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
