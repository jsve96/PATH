import numpy as np
import torch
import os
import sys
import logging

# lib/ files use bare imports; add to path before importing
_lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
sys.path.insert(0, _lib_dir)

from ..data_utils import load_data
from aegan import AeGAN


def main(args):
    data      = load_data(args.dataname)
    test_data = load_data(args.dataname, train=False)
    N, T, d   = data.shape
    M         = test_data.shape[0]

    ckpt_dir = f"models/rtsgan/checkpoints/{args.dataname}"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    logger = logging.getLogger("rtsgan")

    hidden_dim = getattr(args, 'hidden_dim', 24)
    layers     = getattr(args, 'num_layer',  3)
    embed_dim  = hidden_dim * (1 + layers)

    params = {
        "device":         args.device,
        "logger":         logger,
        "root_dir":       ckpt_dir,
        "hidden_dim":     hidden_dim,
        "embed_dim":      embed_dim,
        "layers":         layers,
        "dropout":        0.0,
        "ae_batch_size":  128,
        "gan_batch_size": 512,
        "ae_lr":          1e-3,
        "gan_lr":         1e-4,
        "gan_alpha":      0.99,
        "noise_dim":      getattr(args, 'noise_dim', embed_dim),
        "weight_decay":   0.0,
    }

    model = AeGAN(d, params)
    model.load_ae()
    model.load_generator()

    generated = model.synthesize(M, seq_len=T, batch_size=500)  # (M, T, d)

    np.save(args.save_path, generated)
    print(f"[RTSGAN] Generated {M} samples -> {args.save_path}.npy")
