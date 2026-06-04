import numpy as np
import torch
import os
import sys
import logging

# kovae/models/* uses bare 'models.*' imports, so kovae/ itself must be on path
_kovae_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _kovae_dir)

from ..data_utils import load_data
from .models.kovae import KoVAE


def _set_kovae_args(args, d: int, T: int):
    args.inp_dim      = d
    args.seq_len      = T
    if not hasattr(args, 'num_layers'):
        args.num_layers = getattr(args, 'num_layer', 3)
    args.z_dim        = getattr(args, 'z_dim',        16)
    args.batch_norm   = getattr(args, 'batch_norm',   True)
    args.pinv_solver  = getattr(args, 'pinv_solver',  False)
    args.missing_value= getattr(args, 'missing_value',0.0)
    args.num_steps    = getattr(args, 'num_steps',    1)
    args.w_rec        = getattr(args, 'w_rec',        1.0)
    args.w_kl         = getattr(args, 'w_kl',         0.007)
    args.w_pred_prior = getattr(args, 'w_pred_prior', 0.005)
    args.weight_decay = getattr(args, 'weight_decay', 0.0)
    args.epochs = 600
    args.z_dim = 16
    args.lr = 7e-4


def main(args):
    data      = load_data(args.dataname)
    test_data = load_data(args.dataname, train=False)
    N, T, d   = data.shape
    M         = test_data.shape[0]

    ckpt_dir  = f"models/kovae/checkpoints/{args.dataname}"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    logger = logging.getLogger("kovae")

    _set_kovae_args(args, d, T)

    model = KoVAE(args).to(args.device)
    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    model.eval()
    logger.info(f"Loaded checkpoint from {ckpt_path}")

    generated = []
    remaining = M
    batch_size = 500
    with torch.no_grad():
        while remaining > 0:
            count = min(batch_size, remaining)
            samples = model.sample_data(count).detach().cpu().numpy()
            generated.append(samples)
            remaining -= count

    generated_data = np.concatenate(generated, axis=0)  # (M, T, d)
    np.save(args.save_path, generated_data)
    print(f"[KoVAE] Generated {M} samples -> {args.save_path}.npy")
