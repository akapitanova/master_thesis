import torch.nn as nn
from modules import UNet_conditional
from diffusion import *
from utils import *


def predict(model,
            sampler, 
            test_dl,
            device,
            n_samples=4):
    """
    Return predictions
    """
    x_real = []
    predictions = []

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dl, desc="Testing loop")):
            # for i, data in enumerate(test_dl):
            vectors = data['data'].to(device)
            settings = data['settings'].to(device)

            # with autocast(device_type=device, dtype=torch.float16):
            pred = sampler.ddim_sample_loop(model=model,
                                            y=settings,
                                            cfg_scale=1,
                                            device=device,
                                            eta=1,
                                            n=n_samples
                                            )

            # we move predictions to cpu, in case they are stored on GPU
            x_real.extend(vectors.cpu().tolist())
            predictions.extend(pred.cpu().tolist())

    return x_real, predictions


def evaluate(model,
             sampler,
             device,
             test_csv_path,
             n_samples=4,
             batch_size=4):
    """
    Evaluate predictions
    """
    # Load the test dataset
    x_test, y_test = get_data(test_csv_path)

    test_dataset = CustomDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

    x_real, predictions = predict(model,
                                  sampler,
                                  test_dataloader,
                                  device=device,
                                  n_samples=n_samples)
    mse = nn.MSELoss()
    mse_errors = []

    for i, pred in enumerate(predictions):
        err = mse(pred, x_real[i])
        mse_errors.append(err)

    return mse_errors

def main():
    path = "models/test/ema_ckpt.pt"
    print("Loading ", path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    model = UNet_conditional(length=1024,
                             feat_num=3,
                             device=device).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt)
    sampler = SpacedDiffusion(beta_start=1e-4,
                              beta_end=0.02,
                              noise_steps=1000,
                              section_counts=[40],
                              length=1024,
                              device=device,
                              rescale_timesteps=False)
    mse_errors = evaluate(model,
                          sampler,
                          device,
                          "../data/test_data.csv")

if __name__ == '__main__':
    main()
