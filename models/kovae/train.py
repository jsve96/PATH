import numpy as np
import torch
import torch.utils.data as Data
import torch.optim as optim
import os
import sys
import logging
import time

# kovae/models/* uses bare 'models.*' imports, so kovae/ itself must be on path
_kovae_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _kovae_dir)

from ..data_utils import load_data
from .models.kovae import KoVAE
from .utils.utils import agg_losses, log_losses


def _set_kovae_args(args, d: int, T: int):
    """Attach KoVAE-specific attributes with sensible defaults."""
    args.inp_dim = d
    args.seq_len = T
    # num_layers: KoVAE uses 'num_layers'; pipeline global uses 'num_layer'
    if not hasattr(args, 'num_layers'):
        args.num_layers = getattr(args, 'num_layer', 3)
    args.z_dim        = getattr(args, 'z_dim',        16)
    args.batch_norm   = getattr(args, 'batch_norm',   True)
    args.pinv_solver  = getattr(args, 'pinv_solver',  False)
    args.missing_value= getattr(args, 'missing_value',0.0)
    args.num_steps    = getattr(args, 'num_steps',    1)
    args.w_rec        = getattr(args, 'w_rec',        1.0)
    args.w_kl         = getattr(args, 'w_kl',         0.0009)
    args.w_pred_prior = getattr(args, 'w_pred_prior', 0.009)
    args.weight_decay = getattr(args, 'weight_decay', 0.0)
    print('Set Epochs,z_dim,lr to default for KoVAE')
    args.epochs = 600
    args.z_dim = 16
    args.lr = 7e-4
    if args.dataname == 'stocks':
        print('Use best parameter for stock')
        args.w_kl= .0009
        args.w_pred_prior = .009
    elif args.dataname == 'sine':
        print('Use best parameter for sine')
        args.w_kl= .007
        args.w_pred_prior = .005 


def main(args):
    data = load_data(args.dataname)
    N, T, d = data.shape
    print(f"[KoVAE] data shape: {data.shape}")

    ckpt_dir = f"models/kovae/checkpoints/{args.dataname}"
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
    logger = logging.getLogger("kovae")

    _set_kovae_args(args, d, T)

    epochs     = getattr(args, 'epochs',     600)
    batch_size = getattr(args, 'batch_size', 64)
    lr         = getattr(args, 'lr',         7e-4)

    logger.info(
        f"z_dim={args.z_dim}  hidden_dim={args.hidden_dim}  "
        f"num_layers={args.num_layers}  epochs={epochs}  d={d}  T={T}"
    )

    # DataLoader — no seeding needed here, data is pre-processed
    tensor_data = torch.tensor(data, dtype=torch.float32)
    train_loader = Data.DataLoader(
        Data.TensorDataset(tensor_data),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = KoVAE(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    start_time = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        losses_agg_tr = []
        for (X,) in train_loader:
            X = X.to(args.device)
            optimizer.zero_grad()
            x_rec, Z_enc, Z_enc_prior = model(X)
            batch_losses = model.loss(X, x_rec, Z_enc, Z_enc_prior)
            batch_losses[0].backward()
            optimizer.step()
            losses_agg_tr = agg_losses(losses_agg_tr, batch_losses)

        if (epoch + 1) % 50 == 0:
            log_losses(epoch, losses_agg_tr, model.names)

    runtime_seconds = time.perf_counter() - start_time

    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Checkpoint saved to {ckpt_path}")

    runtime_path = os.path.join(ckpt_dir, "runtime.txt")
    with open(runtime_path, "w") as f:
        f.write(f"runtime_seconds: {runtime_seconds:.4f}\n")
        f.write(f"runtime_minutes: {runtime_seconds / 60:.4f}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"dataset: {args.dataname}\n")
        f.write(f"model: KoVAE\n")

    logger.info(f"Done. Runtime {runtime_seconds:.1f}s -> {ckpt_dir}")
