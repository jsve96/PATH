import numpy as np
import torch
import os
import sys
import logging
import time

# lib/ files use bare imports; add to path before importing
_lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
sys.path.insert(0, _lib_dir)

from ..data_utils import load_data
from aegan import AeGAN


def main(args):
    data = load_data(args.dataname)
    N, T, d = data.shape
    print(f"[RTSGAN] data shape: {data.shape}")

    ckpt_dir = f"models/rtsgan/checkpoints/{args.dataname}"
    os.makedirs(ckpt_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(ckpt_dir, "train.log")),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger("rtsgan")

    # embed_dim is fixed by architecture: hidden_dim * (1 + layers)
    hidden_dim = getattr(args, 'hidden_dim', 24)
    layers     = getattr(args, 'num_layer',  3)
    embed_dim  = hidden_dim * (1 + layers)

    args.iteration = 15000
    args.epochs = 1000

    params = {
        "device":         args.device,
        "logger":         logger,
        "root_dir":       ckpt_dir,
        "hidden_dim":     hidden_dim,
        "embed_dim":      embed_dim,
        "layers":         layers,
        "dropout":        getattr(args, 'dropout',        0.0),
        "ae_batch_size":  getattr(args, 'batch_size',     128),
        "gan_batch_size": getattr(args, 'gan_batch_size',  512),
        "ae_lr":          getattr(args, 'ae_lr',          1e-3),
        "gan_lr":         getattr(args, 'gan_lr',         1e-4),
        "gan_alpha":      getattr(args, 'gan_alpha',      0.99),
        "noise_dim":      getattr(args, 'noise_dim',      embed_dim),
        "weight_decay":   getattr(args, 'weight_decay',   0.0),
    }

    # ae_epochs / gan_iterations: prefer RTSGAN-specific attrs, fall back to
    # shared pipeline args, then use RTSGAN defaults (1000 AE / 15000 GAN).
    ae_epochs      = getattr(args, 'ae_epochs',      getattr(args, 'epochs',    1000))
    gan_iterations = getattr(args, 'gan_iterations', getattr(args, 'iteration', 15000))
    d_update       = getattr(args, 'd_update', 5)

    logger.info(f"hidden_dim={hidden_dim}  layers={layers}  embed_dim={embed_dim}")
    logger.info(f"ae_epochs={ae_epochs}  gan_iterations={gan_iterations}  d={d}  T={T}")

    start_time = time.perf_counter()

    model = AeGAN(d, params)
    model.train_ae(data.astype(np.float32), epochs=ae_epochs)
    model.train_gan(data.astype(np.float32), iterations=gan_iterations, d_update=d_update)

    runtime_seconds = time.perf_counter() - start_time

    runtime_path = os.path.join(ckpt_dir, "runtime.txt")
    with open(runtime_path, "w") as f:
        f.write(f"runtime_seconds: {runtime_seconds:.4f}\n")
        f.write(f"runtime_minutes: {runtime_seconds / 60:.4f}\n")
        f.write(f"ae_epochs: {ae_epochs}\n")
        f.write(f"gan_iterations: {gan_iterations}\n")
        f.write(f"dataset: {args.dataname}\n")
        f.write(f"model: RTSGAN\n")

    logger.info(f"Done. Runtime {runtime_seconds:.1f}s -> {ckpt_dir}")
