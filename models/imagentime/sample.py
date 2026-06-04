import os
import sys
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Project-level data utility — imported before sys.modules eviction so the
# relative `..data_utils` path still resolves against the project's models package.
from ..data_utils import load_data

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR = Path(_HERE) / "configs"

_CONFLICT_PREFIXES = ("models", "utils")


def _evict_project_namespaces():
    evicted = {
        k: v for k, v in list(sys.modules.items())
        if k in _CONFLICT_PREFIXES
        or any(k.startswith(p + ".") for p in _CONFLICT_PREFIXES)
    }
    for k in evicted:
        del sys.modules[k]
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)


def _import_imagentime_internals():
    _evict_project_namespaces()
    from models.model import ImagenTime         # imagentime/models/model.py
    from models.sampler import DiffusionProcess # imagentime/models/sampler.py
    return ImagenTime, DiffusionProcess


_IMAGENTIME_DEFAULTS = {
    "num_workers": 4,
    "resume": False,
    "neptune": False,
    "tags": ["karras", "unconditional"],
    "beta1": 1e-5,
    "betaT": 1e-2,
    "deterministic": False,
    "percent": 100,
    "epochs": 1000,
}


def _load_and_merge_config(args) -> None:
    cfg_path = _CONFIG_DIR / f"{args.dataname}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No imagentime config for '{args.dataname}'. Expected: {cfg_path}"
        )
    config = OmegaConf.to_object(OmegaConf.load(cfg_path))
    for k, v in _IMAGENTIME_DEFAULTS.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
    # dataset config always wins over project-level get_args() defaults
    for k, v in config.items():
        setattr(args, k, v)
    args.dataset = args.dataname


def main(args) -> None:
    _load_and_merge_config(args)
    ImagenTime, DiffusionProcess = _import_imagentime_internals()

    if torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}" if args.gpu is not None else "cuda"
    else:
        args.device = "cpu"

    ckpt_path = Path(_HERE) / "checkpoints" / args.dataname / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run with --mode train first."
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"[imagentime] dataset={args.dataset}  device={args.device}  ckpt={ckpt_path}")

    # number of samples = test-set size
    test_data = load_data(args.dataname, train=False)
    M = test_data.shape[0]
    logging.info(f"Generating {M} samples to match test set.")

    # load model and checkpoint
    model = ImagenTime(args=args, device=args.device).to(args.device)
    ckpt = torch.load(str(ckpt_path), map_location=args.device)
    model.load_state_dict(ckpt["model"], strict=False)
    if "ema_model" in ckpt and args.ema:
        model.model_ema.load_state_dict(ckpt["ema_model"])
        model.model_ema.copy_to(model.net)
        logging.info("EMA weights loaded.")
    model.eval()

    # DelayEmbedder.img_to_ts() needs img_shape, set via a dummy forward pass
    if not args.use_stft:
        dummy = torch.zeros(1, args.seq_len, args.input_channels, device=args.device)
        with torch.no_grad():
            model.ts_to_img(dummy)

    shape = (args.input_channels, args.img_resolution, args.img_resolution)
    process = DiffusionProcess(args, model.net, shape)

    samples = []
    remaining = M
    with torch.no_grad():
        with model.ema_scope():
            while remaining > 0:
                bs = min(args.batch_size, remaining)
                x_img = process.sampling(sampling_number=bs)
                x_ts = model.img_to_ts(x_img)
                samples.append(x_ts.cpu().numpy())
                remaining -= bs
                logging.info(f"  sampled {M - remaining}/{M}")

    synthetic = np.vstack(samples)
    logging.info(f"Output shape: {synthetic.shape}")

    out_path = args.save_path
    if not out_path.endswith(".npy"):
        out_path += ".npy"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, synthetic)
    logging.info(f"Saved → {out_path}")
