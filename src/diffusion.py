import numpy as np
import torch
from tqdm import tqdm
import logging
from torchsummary import summary
import torchvision.transforms.functional as f


# from src.dataset import remove_gain


def prepare_noise_schedule(noise_steps, beta_start, beta_end):
    """
    cosine schedule
    """
    t = torch.linspace(0, 1, noise_steps)
    return beta_end + 0.5 * (beta_start - beta_end) * \
        (1 + torch.cos(t * torch.pi))


class GaussianDiffusion:
    """
     involves initializing, adding noise,
     sampling images, and calculating the mean/variance of the distribution.
    """

    def __init__(self,
                 betas,
                 noise_steps=1000,
                 # img_height=120,
                 # img_width=300,
                 length=1024,
                 device="cuda"):  # beta_start=1e-4, beta_end=0.02
        self.noise_steps = noise_steps

        self.beta = betas.to(device)

        self.alpha = (1. - self.beta).to(device)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(device)

        self.alpha_hat_prev = torch.cat([torch.tensor([1.0], device=device), \
                                         self.alpha_hat[:-1]]).to(device)
        self.alpha_hat_next = torch.cat([self.alpha_hat[1:], \
                                         torch.tensor([0.0], \
                                                      device=device)]).to(device)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # Variance for the posterior distribution during denoising
        self.posterior_variance = (self.beta * (1.0 - self.alpha_hat_prev) /
                                   (1.0 - self.alpha_hat)).to(device)

        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = torch.log(torch.cat(
            [torch.tensor([self.posterior_variance[1]], device=device),
             self.posterior_variance[1:]])).to(device)

        self.sqrt_recip_alpha_hat = torch.sqrt(1.0 / self.alpha_hat).to(device)

        self.sqrt_recipm1_alpha_hat = torch.sqrt(1.0 / self.alpha_hat - 1).to(device)

        self.posterior_mean_coef1 = (self.beta * torch.sqrt(self.alpha_hat_prev) /
                                     (1.0 - self.alpha_hat)).to(device)

        self.posterior_mean_coef2 = ((1.0 - self.alpha_hat_prev) *
                                     torch.sqrt(self.alpha) /
                                     (1.0 - self.alpha_hat)).to(device)

        # self.img_height = img_height
        # self.img_width = img_width
        self.length = length
        self.device = device

    def prepare_noise_schedule(self):
        t = torch.linspace(0, 1, self.noise_steps)
        return self.beta_end + 0.5 * (self.beta_start - self.beta_end) * (1 + torch.cos(t * torch.pi))

    def noise_images(self, x, t, eps=None):
        """
        Adding noise to images.
        This method takes an image tensor x and adds noise to it at timestep t.
        """
        # sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        # sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]

        # Remove channel dimenson
        #x = torch.squeeze(x, 1)

        if eps == None:
            eps = torch.randn_like(x)


        noised_vector = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps

        # Add channel dimension back
        #noised_vector = torch.unsqueeze(noised_vector, 1)
        #eps = torch.unsqueeze(eps, 1)

        return noised_vector, eps

    def sample_timesteps(self, n, all_same):
        """
        Selects timesteps randomly for each sample during the diffusion process.
        """
        if all_same:
            return torch.randint(low=1, high=self.noise_steps, size=(1,)).expand(n)
        else:
            return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample_ddpm(self,
                    model,
                    n,
                    settings,
                    cfg_scale=3,
                    #resize=None
                    ):
        """
        This method uses the DDPM (Denoising Diffusion Probabilistic Model)
        algorithm for sampling by reversing the diffusion process.
        It starts from random noise and progressively removes noise.
        Uses a guidance factor (cfg_scale) for more controlled sampling.
        """
        logging.info(f"Sampling {n} new images....")

        model.eval()

        with torch.no_grad():
            # x = torch.randn((n, 1, self.img_height, self.img_width)).to(self.device)
            x = torch.randn((n, 1, self.length)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, settings)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise,
                                                 predicted_noise,
                                                 cfg_scale)
                # alpha = self.alpha[t][:, None, None, None]
                # alpha_hat = self.alpha_hat[t][:, None, None, None]
                # beta = self.beta[t][:, None, None, None]
                alpha = self.alpha[t][:, None]
                alpha_hat = self.alpha_hat[t][:, None]
                beta = self.beta[t][:, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise

        model.train()

        x = (x.clamp(-1, 1) + 1) / 2
        #x = (x * 255).type(torch.uint8)
        x = (x * 3925).type(torch.uint8)

        #if resize:
        #    x = f.resize(x, resize, antialias=True)
        return x

    def ddim_sample_loop(self,
                         model,
                         y,
                         cfg_scale=3,
                         device=None,
                         eta=0.0,
                         n=4,
                         #resize=(256, 512),
                         # gain=0
                         ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        model.eval()

        for sample in self.ddim_sample_loop_progressive(model,
                                                        y,
                                                        cfg_scale,
                                                        device=device,
                                                        eta=eta,
                                                        n=n):
            final = sample


        # normalization to back numbers, for intensities: lower_bound=0, upper_bound=3925
        #final["sample"] = (final["sample"].clamp(-1, 1) + 1) / 2
        #final["sample"] = (final["sample"] * 255).type(torch.uint8)
        #final["sample"] = (final["sample"] * 3925)
        final["sample"] = (final["sample"].clamp(0, 1))
        final["sample"] = (final["sample"])

        #if resize:
        #    final["sample"] = f.resize(final["sample"], resize, antialias=True)
        # if gain:
        #    final["sample"] = remove_gain(final["sample"], gain, tensor=True)
        return final["sample"].squeeze()

    def ddim_sample_loop_progressive(self,
                                     model,
                                     y,
                                     cfg_scale=3,
                                     device="cpu",
                                     eta=0.0,
                                     n=4):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        indices = list(range(self.noise_steps))[::-1]
        #img = torch.randn((n, 1, self.img_height, self.img_width), device=device)
        img = torch.randn((n, self.length), device=device)
        #for i in tqdm(indices, desc=f"ddim sample loop"):
        for i in indices:
            t = torch.tensor([i] * len(img), device=device)
            with torch.no_grad():
                out = self.ddim_sample(model,
                                       img,
                                       t,
                                       y,
                                       cfg_scale,
                                       eta=eta)
                out['t'] = t
                yield out
                img = out["sample"]

    def ddim_sample(self,
                    model,
                    x,
                    t,
                    y,
                    cfg_scale=3,
                    eta=0.0):  # TODO why is eta 0, isnt nonzero_mask * sigma * noise always 0 then?
        """
        Sample x_{t-1} from the model using DDIM.
        """
        out = self.p_mean_variance(model, x, t.to(self.device), y, cfg_scale)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        # eps = out["model_forward"]

        # alpha_bar = self.alpha_hat[t][:, None, None, None]
        # alpha_bar_prev = self.alpha_hat_prev[t][:, None, None, None]
        alpha_bar = self.alpha_hat[t][:, None]
        alpha_bar_prev = self.alpha_hat_prev[t][:, None]

        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)

        noise = torch.randn_like(x)
        mean_pred = out["pred_xstart"] * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps

        # no noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))

        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_mean_variance(self,
                        model,
                        x,
                        t,
                        y,
                        cfg_scale=3,
                        vartype="fixed_large"):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                        as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :return: a dict with the following keys:
                    - 'mean': the model mean output.
                    - 'variance': the model variance output.
                    - 'log_variance': the log of 'variance'.
                    - 'pred_xstart': the prediction for x_0.
        """
        model_output = model.forward(x, t, y)
        if cfg_scale > 0:
            uncond_model_output = model(x, t, None)
            model_output = torch.lerp(uncond_model_output,
                                      model_output,
                                      cfg_scale)
        if vartype == "fixed_large":
            # print(self.posterior_variance[1])
            model_variance = torch.cat((self.posterior_variance[1].reshape(1),
                                        self.beta[1:]), dim=0)
            model_log_variance = torch.log(model_variance)

        elif vartype == "fixed_small":
            model_variance = self.posterior_variance
            model_log_variance = self.posterior_log_variance_clipped

        # model_variance = model_variance[t][:, None, None, None]
        # model_log_variance = model_log_variance[t][:, None, None, None]
        model_variance = model_variance[t][:, None]
        model_log_variance = model_log_variance[t][:, None]

        pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            'model_forward': model_output,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        # return self.sqrt_recip_alpha_hat[t][:, None, None, None] * x_t - self.sqrt_recipm1_alpha_hat[t][:, None, None, None] * eps
        #x_t = torch.squeeze(x_t, 1)
        #eps = torch.squeeze(eps, 1)
        res = self.sqrt_recip_alpha_hat[t][:, None] * x_t - self.sqrt_recipm1_alpha_hat[t][:, None] * eps
        #res = torch.unsqueeze(res, 1)
        return res

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        # return (self.sqrt_recip_alpha_hat[t][:, None, None, None] * x_t - pred_xstart) / self.sqrt_recipm1_alpha_hat[t][:, None, None, None]
        #x_t = torch.squeeze(x_t, 1)
        #pred_xstart = torch.squeeze(pred_xstart, 1)
        res = (self.sqrt_recip_alpha_hat[t][:, None] * x_t - pred_xstart) / self.sqrt_recipm1_alpha_hat[t][:, None]
        #res = torch.unsqueeze(res, 1)
        return res

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        # posterior_mean = (
        #    self.posterior_mean_coef1[t][:, None, None, None] *
        #    x_start +
        #    self.posterior_mean_coef2[t][:, None, None, None] *
        #    x_t)
        # posterior_variance = self.posterior_variance[t][:, None, None, None]
        # posterior_log_variance_clipped = self.posterior_log_variance_clipped[t][:, None, None, None]

        posterior_mean = (
                self.posterior_mean_coef1[t][:, None] * x_start +
                self.posterior_mean_coef2[t][:, None] * x_t)
        posterior_variance = self.posterior_variance[t][:, None]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t][:, None]

        return posterior_mean, posterior_variance, posterior_log_variance_clipped


class SpacedDiffusion(GaussianDiffusion):
    """
    DDIM
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self,
                 beta_start,
                 beta_end,
                 section_counts,
                 noise_steps=1000,
                 # img_height=120,
                 # img_width=300,
                 length=1024,
                 device="cuda",
                 rescale_timesteps=False):
        # generation of the original noise schedule
        original_betas = prepare_noise_schedule(noise_steps,
                                                beta_start,
                                                beta_end)
        # Create a list of timesteps to use from an original diffusion process
        self.use_timesteps = space_timesteps(num_timesteps=noise_steps,
                                             section_counts=section_counts)  # f'ddim{noise_steps}')

        # how the new t's mapped to the old t's
        self.timestep_map = []
        self.original_num_steps = len(original_betas)
        self.rescale_timesteps = rescale_timesteps

        base_diffusion = GaussianDiffusion(noise_steps=noise_steps,
                                           betas=original_betas,
                                           # img_height=img_height,
                                           # img_width=img_width,
                                           length=length,
                                           device=device)
        last_alpha_hat = 1.0
        # the betas are recalculated for the selected timestamps
        new_betas = []
        for i, alpha_hat in enumerate(base_diffusion.alpha_hat):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_hat / last_alpha_hat)
                last_alpha_hat = alpha_hat
                self.timestep_map.append(i)
        betas = torch.Tensor(new_betas)
        noise_steps = len(self.timestep_map)
        super().__init__(betas,
                         noise_steps,
                         # img_height,
                         # img_width,
                         length,
                         device)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model,
                             self.timestep_map,
                             self.rescale_timesteps,
                             self.original_num_steps)

    def p_mean_variance(self, model, *args, **kwargs):  # pylint: disable=signature-differs
        """
        This method computes the mean and variance at each
        timestep for the reverse diffusion process.
        It’s overridden here to wrap the model correctly
        using self._wrap_model(model).
        """
        return super().p_mean_variance(self._wrap_model(model), *args,
                                       **kwargs)


class _WrappedModel:
    """
    Converting the supplied t's to the old t's scales.
    Adjusts the input timesteps (t) to the original timesteps,
    ensuring that the model sees the correct range of diffusion steps,
    even when only a subset of them are used. The SpacedDiffusion class
    uses this wrapper to remap the timesteps before passing them to the model.
    """

    def __init__(self,
                 model,
                 timestep_map,
                 rescale_timesteps,
                 original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def forward(self, x, t, y):
        """
        Args:
            t: t's with differrent ranges (can be << T due to smaller eval T)
              need to be converted to the original t's
        """
        map_tensor = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)

        def do(t):
            new_ts = map_tensor[t]
            if self.rescale_timesteps:
                new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
            return new_ts

        return self.model(x, do(t), y)

    def __call__(self, x, t, y):
        return self.forward(x, t, y)

    def __getattr__(self, name):
        if hasattr(self.model, name):
            func = getattr(self.model, name)
            return func
        raise AttributeError(name)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}")
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


    #model, timestep_map, rescale_timesteps, original_num_steps

class EdmSampler:
    """
    Proposed EDM sampler (Algorithm 2).
    """
    def __init__(
            self,
            net,
            num_steps=18,
            sigma_min=0.002,
            sigma_max=80,
            rho=7,
            S_churn=0,
            S_min=0,
            S_max=float('inf'),
            S_noise=1,
            randn_like=torch.randn_like
    ):
        """
        Initializes the EDM Sampler.

        Attributes:
        - net: Neural network used for denoising and sigma adjustment.
        - num_steps: Number of discretization steps for the diffusion process.
        - sigma_min: Minimum noise level for the diffusion process.
        - sigma_max: Maximum noise level for the diffusion process.
        - rho: Controls the distribution of timesteps (exponential factor).
        - S_churn: Factor for temporary noise level increase.
        - S_min: Minimum noise level for applying S_churn.
        - S_max: Maximum noise level for applying S_churn.
        - S_noise: Noise magnitude for temporary increase in the process.
        - randn_like: Function for generating random noise.
        """
        self.net = net
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.randn_like = randn_like

    def sample(self, length, device, class_labels=None, n_samples=1):
        """
        Runs the diffusion sampling process.

        Args:
        - length: Length of the latent vectors.
        - device: Device to run the sampling on.
        - class_labels: Optional class labels for conditional generation.
        - n_samples: Number of samples to generate in parallel.

        Returns:
        - Final sampled tensor after the diffusion process.
        """
        # Create initial latents.
        latents = self.randn_like(torch.empty((n_samples, length), device=device))

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(self.sigma_min, self.net.sigma_min)
        sigma_max = min(self.sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=device)
        t_steps = (
            sigma_max ** (1 / self.rho)
            + step_indices / (self.num_steps - 1) * (sigma_min ** (1 / self.rho)
                                                     - sigma_max ** (1 / self.rho))
        ) ** self.rho
        # t_N = 0
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * self.randn_like(x_cur)

            # Euler step.
            denoised = self.net(x_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.num_steps - 1:
                denoised = self.net(x_next, t_next, class_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next
