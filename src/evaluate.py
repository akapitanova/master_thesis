import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from modules import UNet_conditional, EDMPrecond, MaxScalePredictor
from diffusion import GaussianDiffusion, SpacedDiffusion, EdmSampler
from utils import *
from dataset import CustomDataset, get_data

def normalize_sampled_vectors(vectors, device, predicted_max_scale):
    """Normalizes input vectors between -1 and 1."""
    lower_bound = vectors.min(dim=-1, keepdim=True)[0]
    upper_bound = vectors.max(dim=-1, keepdim=True)[0]
    norm_min = torch.tensor([-1], device=device).reshape(-1, 1, 1)
    norm_max = predicted_max_scale.reshape(-1, 1, 1)
    return norm_min + (vectors - lower_bound) * (norm_max - norm_min) / (upper_bound - lower_bound)

def save_predictions(x_real, cond_vectors, predictions, predictions_path):
    """
    Saves the predictions into a CSV file.

    Parameters:
        x_real (list): List of real input vectors.
        cond_vectors (list): List of conditioning vectors.
        predictions (list or numpy array): Model-generated predictions.
        predictions_path (str): Path to save the CSV file.
    """
    # Convert lists of lists into CSV-friendly string format
    x_real_str = [','.join(map(str, row)) for row in x_real]
    cond_vectors_str = [','.join(map(str, row)) for row in cond_vectors]
    preds_str = [','.join(map(str, row)) for row in predictions]
    # Ensure predictions is in the correct format before slicing
    #predictions = torch.tensor(predictions) if isinstance(predictions, list) else predictions
    #preds_str = [','.join(map(str, row)) for row in predictions[:, 0, 0, :].tolist()]
    #preds_str = [','.join(map(str, row)) for row in predictions.tolist()]
    
    # Create and save dataframe
    df = pd.DataFrame({'x_real': x_real_str, 'cond_vectors': cond_vectors_str, 'predictions': preds_str})
    df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
    return df


def predict(model,
            scale_predictor,
            sampler,
            test_dl,
            device,
            n_samples=1,
            s_type='edm',
            cfg_scale=1,
            settings_dim=13):
    """
    Return predictions using the specified sampler.
    """
    x_real = np.empty((0,))  
    cond_vectors = np.empty((0,))
    predictions = np.empty((0,))  

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dl, desc="Testing loop")):
            vectors = data['data'].to(device)
            resolution = vectors.size(1)
            settings = data['settings'].to(device)
 
            if s_type == 'ddim':
                pred = sampler.ddim_sample_loop(
                                                model=model,
                                                y=settings,
                                                cfg_scale=cfg_scale,
                                                device=device,
                                                eta=1,
                                                n=n_samples,
                                                )
            elif s_type == 'edm':
                pred = sampler.sample(
                                    resolution=resolution,
                                    device=device,
                                    settings=settings,
                                    n_samples=n_samples,
                                    cfg_scale=cfg_scale,
                                    settings_dim=settings_dim
                                    )

            predicted_max_scale = scale_predictor(settings).detach()
            
            pred = normalize_sampled_vectors(pred, device, predicted_max_scale)

            # Convert to NumPy and repeat where necessary
            vectors_np = vectors.cpu().numpy().repeat(n_samples, axis=0)
            settings_np = settings.cpu().numpy().repeat(n_samples, axis=0)
            pred_np = pred.cpu().numpy()
    
            # Append using np.concatenate
            x_real = np.concatenate((x_real, vectors_np), axis=0) if x_real.size else vectors_np
            cond_vectors = np.concatenate((cond_vectors, settings_np), axis=0) if cond_vectors.size else settings_np
            predictions = np.concatenate((predictions, pred_np), axis=0) if predictions.size else pred_np
    return x_real, cond_vectors, predictions[:, 0, :]


def evaluate(device,
             test_csv_path,
             model_path,
             scale_predictor_path,
             n_samples=1,
             s_type='edm',
             settings_dim=13,
             cfg_scale=2,
            noise_steps=100,
            section_counts=40):
    """
    Evaluate predictions
    """
    # Load the model and sampler
    if s_type == 'edm':
        model = EDMPrecond(device=device).to(device)
        ckpt = torch.load(model_path,
                          map_location=device,
                          weights_only=True)
        model.load_state_dict(ckpt)
        
        sampler = EdmSampler(net=model, num_steps=noise_steps)

    elif s_type == 'ddim':
        model = UNet_conditional(device=device, sampler_type="DDIM").to(device)
        ckpt = torch.load(model_path,
                          map_location=device,
                          weights_only=True
                          )
        model.load_state_dict(ckpt)
        
        sampler = SpacedDiffusion(beta_start=1e-4,
                                  beta_end=0.02,
                                  section_counts=[section_counts],
                                  device=device)
        
    else:
        raise ValueError("Unknown model type. Please use 'ddim' or 'edm'.")

    # Load model for scale prediction
    scale_predictor = MaxScalePredictor().to(device)
    ckpt_sp = torch.load(scale_predictor_path,
                         map_location=device,
                         weights_only=True
                        )
    scale_predictor.load_state_dict(ckpt_sp)
    
    # Load the test dataset
    x_test, y_test = get_data(test_csv_path)

    test_dataset = CustomDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False)

    x_real, cond_vectors, predictions = predict(model,                      
                                                scale_predictor,
                                                sampler,
                                                test_dataloader,
                                                device=device,
                                                n_samples=n_samples,
                                                s_type=s_type,
                                                cfg_scale=cfg_scale,
                                                settings_dim=settings_dim
                                                )

    return x_real, cond_vectors, predictions

def main():
    parser = argparse.ArgumentParser(description="Evaluate a diffusion model with specified parameters.")
    
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the evaluation (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--test_csv_path", type=str, required=True, help="Path to the test dataset CSV file.")
    parser.add_argument("--predictions_path", type=str, required=True, help="Path to save the predictions CSV file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--scale_predictor_path", type=str, required=True, help="Path to the scale predictor model checkpoint.")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate per test instance.")
    parser.add_argument("--s_type", type=str, choices=["edm", "ddim"], default="edm", help="Sampling type: 'edm' or 'ddim'.")
    parser.add_argument("--settings_dim", type=int, default=13, help="Dimensionality of conditioning settings.")
    parser.add_argument("--cfg_scale", type=float, default=2.0, help="Classifier-free guidance scale.")
    parser.add_argument("--noise_steps", type=int, default=100, help="Number of noise steps for sampling.")
    parser.add_argument("--section_counts", type=int, default=40, help="Number of section counts for DDIM sampling.")

    args = parser.parse_args()

    x_real, cond_vectors, predictions = evaluate(
        device=args.device,
        test_csv_path=args.test_csv_path,
        model_path=args.model_path,
        scale_predictor_path=args.scale_predictor_path,
        n_samples=args.n_samples,
        s_type=args.s_type,
        settings_dim=args.settings_dim,
        cfg_scale=args.cfg_scale,
        noise_steps=args.noise_steps,
        section_counts=args.section_counts
    )

    save_predictions(x_real, cond_vectors, predictions, args.predictions_path)

if __name__ == "__main__":
    main()

