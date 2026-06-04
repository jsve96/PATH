"""
Unconditional synthetic sample generation from a trained ImagenTime checkpoint.
Works for any dataset the model was trained on.

Usage:
    python generate_unconditional.py \
        --checkpoint ./logs/stock/conditional-bs=128-.../checkpoint.pt \
        --config     ./configs/unconditional/stocks.yaml \
        --n_samples  1000 \
        --output     ./outputs/synthetic_stock.npy

For datasets with a CSV source (stock, energy), samples are inverse-scaled back to
the original value range by default.  Pass --no_inverse to keep the [0,1] output.

Supported datasets:
    short-range  : stock, energy, sine, mujoco
    long-range   : fred_md, nn5_daily, solar_weekly, temperature_rain,
                   traffic_hourly, kdd_cup
"""

import os
import sys
import yaml
import argparse
import logging
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from models.model import ImagenTime
from models.sampler import DiffusionProcess
from utils.utils_data import gen_dataloader


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Datasets whose raw CSV allows exact inverse MinMax scaling
_CSV_DATASETS = {
    "stock":  "./data/short_range/stock_data.csv",
    "energy": "./data/short_range/energy_data.csv",
}

# Datasets that are already normalized and have no meaningful inverse transform
_NO_INVERSE_DATASETS = {"sine", "mujoco"}

# Datasets whose generated output should be clamped to [0,1]
_CLAMP_DATASETS = {"temperature_rain"}

SHORT_RANGE = {"stock", "energy", "sine", "mujoco"}
LONG_RANGE   = {"fred_md", "nn5_daily", "solar_weekly", "temperature_rain",
                "traffic_hourly", "kdd_cup"}
ALL_DATASETS = SHORT_RANGE | LONG_RANGE


# ---------------------------------------------------------------------------
# Scaling helpers
# ---------------------------------------------------------------------------

def _load_csv_scalers(csv_path: str):
    """Return per-column (min, max) from the original CSV (pre-normalization)."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    data = data[::-1]  # chronological flip, same as training
    return np.min(data, axis=0), np.max(data, axis=0)


def _inverse_minmax(data: np.ndarray, col_min: np.ndarray, col_max: np.ndarray) -> np.ndarray:
    return data * (col_max - col_min + 1e-7) + col_min


def maybe_inverse_scale(samples: np.ndarray, dataset: str, data_dir: str, no_inverse: bool) -> np.ndarray:
    """Apply post-processing specific to each dataset type."""
    if dataset in _CLAMP_DATASETS:
        samples = np.clip(samples, 0.0, 1.0)

    if no_inverse or dataset not in _CSV_DATASETS:
        return samples

    csv_path = os.path.join(data_dir, os.path.basename(_CSV_DATASETS[dataset]))
    if not os.path.exists(csv_path):
        # fall back to the hard-coded default path
        csv_path = _CSV_DATASETS[dataset]
    if not os.path.exists(csv_path):
        logging.warning(f"CSV not found at {csv_path}; skipping inverse scaling.")
        return samples

    col_min, col_max = _load_csv_scalers(csv_path)
    samples = _inverse_minmax(samples, col_min, col_max)
    logging.info(f"Inverse MinMax applied using {csv_path}")
    return samples


# ---------------------------------------------------------------------------
# Args / config
# ---------------------------------------------------------------------------

def build_args(config_path: str, checkpoint: str, device: str) -> argparse.Namespace:
    cfg = OmegaConf.to_object(OmegaConf.load(config_path))
    args = argparse.Namespace(**cfg)

    args.device    = device
    args.log_dir   = checkpoint
    args.seed      = 0
    args.beta1     = getattr(args, "beta1", 1e-5)
    args.betaT     = getattr(args, "betaT", 1e-2)
    args.deterministic = False
    args.num_workers   = 4
    args.neptune   = False
    args.resume    = False
    args.percent   = 100
    args.logging_iter = 100
    args.tags      = [args.dataset, "inference"]

    # short-range datasets need input_size set
    if args.dataset in SHORT_RANGE:
        args.input_size = args.input_channels

    return args


# ---------------------------------------------------------------------------
# Model loading & generation
# ---------------------------------------------------------------------------

def _init_embedder_shape(model: ImagenTime, args: argparse.Namespace) -> None:
    """
    DelayEmbedder.img_to_ts() needs self.img_shape, which is only set after a
    ts_to_img() call.  Bootstrap it once with a dummy forward pass so that
    pure-inference runs work without loading any real data.
    """
    if args.use_stft:
        return  # STFTEmbedder does not use img_shape
    dummy = torch.zeros(1, args.seq_len, args.input_channels, device=args.device)
    model.ts_to_img(dummy)


@torch.no_grad()
def generate(args: argparse.Namespace, n_samples: int, batch_size: int) -> np.ndarray:
    model = ImagenTime(args=args, device=args.device).to(args.device)

    ckpt = torch.load(args.log_dir, map_location=args.device)
    model.load_state_dict(ckpt["model"], strict=False)

    if "ema_model" in ckpt and args.ema:
        model.model_ema.load_state_dict(ckpt["ema_model"])
        model.model_ema.copy_to(model.net)
        logging.info("EMA weights loaded.")
    
    if args.use_stft:
            train_loader, test_loader = gen_dataloader(args)
            model.init_stft_embedder(train_loader)

    model.eval()
    _init_embedder_shape(model, args)

    shape   = (args.input_channels, args.img_resolution, args.img_resolution)
    process = DiffusionProcess(args, model.net, shape)

    samples   = []
    remaining = n_samples
    with tqdm(total=n_samples, desc=f"Generating [{args.dataset}]") as pbar:
        while remaining > 0:
            bs    = min(batch_size, remaining)
            x_img = process.sampling(sampling_number=bs)   # (bs, C, H, W)
            print(x_img.shape)
            x_ts  = model.img_to_ts(x_img)                 # (bs, seq_len, features)
            samples.append(x_ts.cpu().numpy())
            remaining -= bs
            pbar.update(bs)

    return np.vstack(samples)                              # (n_samples, seq_len, features)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic time-series samples from a trained ImagenTime checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the saved .pt checkpoint file")
    parser.add_argument("--config", required=True,
                        help="Path to the YAML config used for training "
                             "(e.g. ./configs/unconditional/stocks.yaml)")
    parser.add_argument("--n_samples",  type=int, default=1000,
                        help="Number of synthetic sequences to generate")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for the sampling loop")
    parser.add_argument("--output",     default="./outputs/synthetic.npy",
                        help="Output path for the generated .npy array")
    parser.add_argument("--data_dir",   default="./data/short_range",
                        help="Directory containing the original CSV files "
                             "(used for inverse MinMax scaling of stock / energy)")
    parser.add_argument("--no_inverse", action="store_true",
                        help="Skip inverse scaling — output stays in [0, 1]")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device string, e.g. 'cpu', 'cuda', 'cuda:1'")
    cli = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    args = build_args(cli.config, cli.checkpoint, cli.device)
    logging.info(f"Dataset : {args.dataset}")
    logging.info(f"Device  : {cli.device}")

    ## load SAMPLE FILE
    with open("configs/SAMPLES.yaml") as f:
        config_samples = yaml.safe_load(f)

    logging.info(f"Samples : {cli.n_samples}")


    if args.dataset == 'stock':
        args.dataset = 'stocks'
    synthetic = generate(args, n_samples=config_samples[args.dataset], batch_size=cli.batch_size)
    logging.info(f"Raw output shape : {synthetic.shape}  (n_samples, seq_len, features)")

    synthetic = maybe_inverse_scale(synthetic, args.dataset, cli.data_dir, cli.no_inverse)

    os.makedirs(os.path.dirname(os.path.abspath(cli.output)), exist_ok=True)
    np.save(cli.output, synthetic)
    logging.info(f"Saved → {cli.output}")
    logging.info(f"  shape={synthetic.shape}  min={synthetic.min():.4f}  "
                 f"max={synthetic.max():.4f}  mean={synthetic.mean():.4f}")


if __name__ == "__main__":
    main()
