# Hyperparameters

All methods are invoked via `python main.py --dataname <dataset> --method <method> --mode train`.
Method-specific arguments are passed on the same command line.

---

## PATH

Non-parametric kernel-smoothed path estimator. No training; runs in a single forward pass.

| Argument | Default | Description |
|---|---|---|
| `--K` | `1` | Markov order (conditioning window length) |
| `--h` | `0.05` | Kernel bandwidth |
| `--kernel` | `0` (Quartic) | Kernel type: `0` Quartic, `1` Gaussian, `2` Epanechnikov, `3` Triangular |

---

## PATH-ADA

Adaptive-bandwidth variant of PATH. Selects the bandwidth automatically from the nearest-neighbour fraction `q`.

| Argument | Default | Description |
|---|---|---|
| `--K` | `1` | Markov order |
| `--q` | `0.001` | Neighbour fraction used for adaptive bandwidth selection (0–1 for fraction, >1 for absolute count) |
| `--kernel` | `0` (Quartic) | Kernel type: `0` Quartic, `1` Gaussian, `2` Epanechnikov, `3` Triangular |

---

## SBTS

Signature-based model. No learnable parameters; fixed simulation algorithm.

| Argument | Default | Description |
|---|---|---|
| `--K` | `1` | Markov order |
| `--h` | `0.05` | Kernel bandwidth |
| `--N_pi` | `100` | Number of Monte Carlo particles per simulation step |

---

## TimeGAN

Recurrent GAN with embedding, supervisor, generator, and discriminator networks.

| Argument | Default | Description |
|---|---|---|
| `--hidden_dim` | `24` | Hidden state size of all RNN components |
| `--num_layer` | `3` | Number of RNN layers |
| `--module` | `gru` | RNN cell type: `gru`, `lstm`, `lstmLN` |
| `--iteration` | `50000` | Total training iterations |
| `--batch_size` | `128` | Mini-batch size |
| `--lr` | `0.001` | Adam learning rate |
| `--beta1` | `0.9` | Adam β₁ momentum |
| `--w_gamma` | `1` | Unsupervised loss weight γ |
| `--w_es` | `0.1` | Supervised encoder loss weight |
| `--w_e0` | `10` | Embedding reconstruction loss weight |
| `--w_g` | `100` | Generator loss weight |

---

## TimeVAE

Variational autoencoder with dense encoder/decoder and optional trend/seasonality components.
Hyperparameters are read from `models/timevae/src/config/hyperparameters.yaml` (variant `timevae`).

| Argument | Default | Description |
|---|---|---|
| `latent_dim` | `8` | Latent space dimensionality |
| `hidden_layer_sizes` | `[50, 100, 200]` | Encoder/decoder MLP hidden widths |
| `reconstruction_wt` | `3.0` | Weight on reconstruction term in ELBO |
| `batch_size` | `16` | Training batch size |
| `use_residual_conn` | `true` | Residual connections in decoder |
| `trend_poly` | `0` | Degree of polynomial trend component (0 = disabled) |
| `custom_seas` | `null` | Custom seasonality period (null = disabled) |
| epochs (hardcoded) | `200` | Training epochs (`train.py` line 18) |

---

## TSDiff

Score-based diffusion model for time series (DSPD / CSPD variants).

| Argument | Default | Description |
|---|---|---|
| `--diffusion` | — | Diffusion type: `GaussianDiffusion`, `OUDiffusion`, `GPDiffusion`, `ContinuousGaussianDiffusion`, `ContinuousOUDiffusion`, `ContinuousGPDiffusion` |
| `--model` | `rnn` | Backbone: `feedforward`, `rnn`, `cnn`, `transformer`, `ode` |
| `--discrete_num_steps` | `1000` | Number of diffusion steps (discrete variants; 100–200 typically sufficient) |
| `--epochs` | `100` | Maximum training epochs |
| `--patience` | `20` | Early-stopping patience; set equal to `--epochs` to disable |
| `--gp_sigma` | `0.1` | GP kernel bandwidth (only for `GPDiffusion` / `ContinuousGPDiffusion`) |
| `--ou_theta` | `0.5` | OU mean-reversion rate (only for `OUDiffusion` / `ContinuousOUDiffusion`) |

Variant mapping: DSPD-GP → `GaussianDiffusion` + GP prior, CSPD-GP → `ContinuousGaussianDiffusion` + GP prior, etc.

---

## RTSGAN

Two-stage model: recurrent autoencoder (AE) pre-trained first, then a GAN trained in the embedding space.

| Argument | Default | Description |
|---|---|---|
| `--hidden_dim` | `24` | RNN hidden size (also determines `embed_dim = hidden_dim × (1 + num_layer)`) |
| `--num_layer` | `3` | Number of RNN layers in AE |
| `ae_epochs` | `1000` | Autoencoder pre-training epochs |
| `gan_iterations` | `15000` | GAN training iterations (uses `--iteration` if no dedicated flag) |
| `ae_lr` | `1e-3` | AE optimizer learning rate |
| `gan_lr` | `1e-4` | GAN optimizer learning rate |
| `gan_alpha` | `0.99` | Generator EMA decay |
| `gan_batch_size` | `512` | GAN mini-batch size |
| `--batch_size` | `128` | AE mini-batch size |
| `dropout` | `0.0` | Dropout in AE |
| `d_update` | `5` | Discriminator update steps per generator step |
| `noise_dim` | `embed_dim` | Noise vector dimension fed to GAN generator |

---

## KoVAE

Koopman operator–based VAE with linear latent dynamics.

| Argument | Default | Description |
|---|---|---|
| `--z_dim` | `16` | Latent dimension |
| `--hidden_dim` | `24` | Encoder/decoder hidden size |
| `--num_layer` / `num_layers` | `3` | Number of layers |
| `--epochs` | `600` | Training epochs (overridden to 600 internally) |
| `--batch_size` | `64` | Mini-batch size |
| `--lr` | `7e-4` | Adam learning rate (overridden to 7e-4 internally) |
| `w_rec` | `1.0` | Reconstruction loss weight |
| `w_kl` | `0.0009` | KL divergence weight (0.007 for `sine`) |
| `w_pred_prior` | `0.009` | Predictive prior loss weight (0.005 for `sine`) |
| `num_steps` | `1` | Koopman prediction horizon steps |
| `batch_norm` | `true` | Batch normalisation in encoder/decoder |

---

## ImagenTime

U-Net diffusion model operating on delay-embedded or STFT images of time series.
All hyperparameters are set per dataset via YAML configs in `models/imagentime/configs/`.

### Shared training hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `epochs` | `1000` | Training epochs |
| `batch_size` | `128` | Mini-batch size (8 for `traffic_hourly`) |
| `learning_rate` | `1e-4` | AdamW learning rate |
| `weight_decay` | `1e-5` | AdamW weight decay |
| `diffusion_steps` | `18` | Number of denoising steps |
| `ema` | `true` | Exponential moving average of weights |
| `ema_warmup` | `100` | Epochs before EMA activates |
| `logging_iter` | `100` | Evaluation frequency (epochs) |

### Per-dataset config (`delay` embedding)

| Dataset | `seq_len` | `use_stft` | `delay` | `embedding` | `img_resolution` | `input_channels` | `unet_channels` | `ch_mult` |
|---|---|---|---|---|---|---|---|---|
| `ecg` | 140 | false | 9 | 16 | 16 | 1 | 128 | [1,2,2,4] |
| `eye` | 128 | false | 16 | 16 | 16 | 14 | 128 | [1,2,2,4] |
| `energy` | 24 | false | 3 | 8 | 8 | 28 | 128 | [1,2,2,4] |
| `stocks` | 24 | false | 3 | 8 | 8 | 6 | 128 | [1,2,2,2] |
| `mujoco` | 24 | false | 3 | 8 | 8 | 14 | 64 | [1,2,2,2] |
| `sine` | 24 | false | 3 | 8 | 8 | 5 | 128 | [1,2,2,2] |
| `traffic_hourly` | (full) | false | 136 | 144 | 144 | 1 | 128 | [1,2,4,4] |

**Delay embedding**: `ts_to_img` tiles windows of length `embedding` with stride `delay` into a `(channels × embedding × n_windows)` image that is then padded to `img_resolution × img_resolution`.
