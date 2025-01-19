import torch
import torch.nn as nn
import torch.nn.functional as F

class EMA:
  """
  Exponential Moving Average (EMA) class
  helps smooth out the model's parameters during training,
  which can lead to a more stable and better-performing
  model by averaging recent updates with past parameter values.

  Exponential Moving Average it's a technique used to make results
  better and more stable training. It works by keeping a copy of
  the model weights of the previous iteration
  and updating the current iteration weights by a factor of (1-beta).
  """
  def __init__(self, beta):
      super().__init__()
      self.beta = beta
      self.step = 0

  def update_model_average(self, ma_model, current_model):
      for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
          old_weight, up_weight = ma_params.data, current_params.data
          ma_params.data = self.update_average(old_weight, up_weight)

  def update_average(self, old, new):
      if old is None:
          return new
      return old * self.beta + (1 - self.beta) * new

  def step_ema(self, ema_model, model, step_start_ema=2000):
      if self.step < step_start_ema:
          self.reset_parameters(ema_model, model)
          self.step += 1
          return
      self.update_model_average(ema_model, model)
      self.step += 1

  def reset_parameters(self, ema_model, model):
      ema_model.load_state_dict(model.state_dict())

class SelfAttention(nn.Module):
  def __init__(self,
                channels,
                length
                ):
      super(SelfAttention, self).__init__()
      self.channels = channels
      self.length = length
      self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
      self.ln = nn.LayerNorm([channels])
      self.ff_self = nn.Sequential(
          nn.LayerNorm([channels]),
          nn.Linear(channels, channels),
          nn.GELU(),
          nn.Linear(channels, channels),
      )

  def forward(self, x):
      # Reshape to (batch, length, channels) for 1D compatibility
      x = x.view(-1, self.length, self.channels)
      x_ln = self.ln(x)
      attention_value, _ = self.mha(x_ln, x_ln, x_ln)
      attention_value = attention_value + x
      attention_value = self.ff_self(attention_value) + attention_value
      # Reshape back to (batch, channels, length)
      return attention_value.view(-1, self.channels, self.length)

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
      super().__init__()
      self.residual = residual
      if not mid_channels:
          mid_channels = out_channels

      # Changed Conv2d to Conv1d for 1D data processing
      self.double_conv = nn.Sequential(
          nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
          nn.GroupNorm(1, mid_channels),
          nn.GELU(),
          nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
          nn.GroupNorm(1, out_channels),
      )


  def forward(self, x):
      if self.residual:
          return F.gelu(x + self.double_conv(x))
      else:
          return self.double_conv(x)

class Down(nn.Module):
  def __init__(self, in_channels, out_channels, emb_dim=256):
      super().__init__()

      self.maxpool_conv = nn.Sequential(
          # Changed MaxPool2d to MaxPool1d for 1D data
          nn.MaxPool1d(2),
          #    nn.MaxPool2d(2),
          DoubleConv(in_channels, in_channels, residual=True),
          DoubleConv(in_channels, out_channels),
          # nn.Dropout(p=0.2),
      )

      self.emb_layer = nn.Sequential(
          nn.SiLU(),
          nn.Linear(
              emb_dim,
              out_channels
          ),
          # nn.Dropout(p=0.2),
      )

  def forward(self, x, t):
      x = self.maxpool_conv(x)
      # Embedding repeated along the new length dimension for 1D data
      emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
      return x + emb

class Up(nn.Module):
  def __init__(self, in_channels, out_channels, emb_dim=256):
      super().__init__()

      # Changed Upsample mode to "linear" for 1D data
      self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
      self.conv = nn.Sequential(
          DoubleConv(in_channels, in_channels, residual=True),
          DoubleConv(in_channels, out_channels, in_channels // 2),
          # nn.Dropout(p=0.2),
      )

      self.emb_layer = nn.Sequential(
          nn.SiLU(),
          nn.Linear(
              emb_dim,
              out_channels
          ),
          # nn.Dropout(p=0.2),
      )

  def forward(self, x, skip_x, t):
      x = self.up(x)
      x = torch.cat([skip_x, x], dim=1)
      x = self.conv(x)
      # Embedding repeated along the new length dimension for 1D data
      emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
      return x + emb


class UNet_conditional(nn.Module):

    def __init__(self,
                 c_in=1,
                 c_out=1,
                 time_dim=256,
                 device="cuda",
                 feat_num=3,
                 length=1024,):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        # Updated SelfAttention dimensions for 1D input (length // 2)
        self.sa1 = SelfAttention(128, length // 2)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, length // 4)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, length // 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, length // 4)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, length // 2)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, length)

        self.outc = nn.Conv1d(64, c_out, kernel_size=1)

        self.label_prep = nn.Sequential(
           nn.BatchNorm1d(feat_num),
           nn.Linear(feat_num, time_dim),
           nn.SiLU(),
        )

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        ).to(self.device)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # Add class embedding to time encoding if class label y is provided
        if y is not None:
            y = self.label_prep(y).squeeze()
            t += y


        # TODO problem for ddim
        #x = torch.unsqueeze(x, 1)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)

        output = torch.squeeze(output, 1)
        return output

#----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

class EDMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'UNet_conditional',   # Class name of the underlying model.
        device          = 'cuda' 
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = globals()[model_type](c_in = img_channels,
                                            c_out = img_channels,
                                            time_dim = 256,
                                            #device = "cuda",
                                            device=device,
                                            feat_num = label_dim,
                                            length = img_resolution)


    def forward(self,
                x,
                sigma,
                class_labels=None
                ):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1)
        class_labels = None if self.label_dim == 0 or class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype),
                         c_noise.flatten(),
                         class_labels)
        assert F_x.dtype == dtype
        F_x = torch.unsqueeze(F_x, 1)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------
