### PATH / SBTS

| Argument | Default | Description |
|---|---|---|
| `--K` | `1` | Markov order |
| `--h` | `0.05` | Kernel bandwidth |

### TimeGAN

| Argument | Default | Description |
|---|---|---|
| `--hidden_dim` | `24` | Hidden state size |
| `--iteration` | `50000` | Training iterations |
| `--module` | `gru` | RNN cell: `gru`, `lstm`, `lstmLN` |

### TSDiff

| Argument | Default | Description |
|---|---|---|
| `--diffusion` | — | `GaussianDiffusion`, `OUDiffusion`, `GPDiffusion`, `ContinuousGaussianDiffusion`, `ContinuousOUDiffusion`, `ContinuousGPDiffusion` |
| `--model` | `rnn` | `feedforward`, `rnn`, `cnn`, `transformer`, `ode` |
| `--discrete_num_steps` | `1000` | Diffusion steps (discrete variants); 100–200 converges faster |
| `--epochs` | `100` | Max training epochs |
| `--patience` | `20` | Early stopping patience; set equal to `--epochs` to disable |
| `--gp_sigma` | `0.1` | GP kernel bandwidth (`GPDiffusion` only) |
| `--ou_theta` | `0.5` | OU mean-reversion (`OUDiffusion` only) |

Corresponds to DSPD-GP, DSPD-OU, CSPD-GP, CSPD-OU
