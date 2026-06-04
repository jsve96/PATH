import os
import sys
import time
import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing
from omegaconf import OmegaConf

torch.multiprocessing.set_sharing_strategy("file_system")

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR = Path(_HERE) / "configs"

# Imagentime uses flat imports (from models.X, from utils.X, from metrics.X)
# that are relative to the imagentime/ directory, but the project already has
# top-level `models` and `utils` packages in sys.modules.  We must NOT import
# imagentime internals at module level; instead we do it lazily inside main()
# after evicting the conflicting entries from sys.modules.

_CONFLICT_PREFIXES = ("models", "utils")


def _evict_project_namespaces():
    """Remove project-level models.* and utils.* from sys.modules so that
    imagentime's flat imports resolve against imagentime/ instead."""
    evicted = {
        k: v for k, v in list(sys.modules.items())
        if k in _CONFLICT_PREFIXES
        or any(k.startswith(p + ".") for p in _CONFLICT_PREFIXES)
    }
    for k in evicted:
        del sys.modules[k]
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    return evicted


def _import_imagentime_internals():
    _evict_project_namespaces()
    # All imports below resolve against imagentime/ because _HERE is first on
    # sys.path and sys.modules no longer has conflicting project entries.
    from models.model import ImagenTime            # imagentime/models/model.py
    from models.sampler import DiffusionProcess    # imagentime/models/sampler.py
    from metrics import evaluate_model_uncond      # imagentime/metrics/__init__.py
    from utils.loggers import PrintLogger          # imagentime/utils/loggers/
    from utils.utils import (                      # imagentime/utils/utils.py
        save_checkpoint, restore_checkpoint, print_model_params,
    )
    from utils.utils_data import gen_dataloader    # imagentime/utils/utils_data.py
    return (
        ImagenTime, DiffusionProcess, evaluate_model_uncond,
        PrintLogger, save_checkpoint, restore_checkpoint,
        print_model_params, gen_dataloader,
    )


_IMAGENTIME_DEFAULTS = {
    "num_workers": 4,
    "resume": False,
    "neptune": False,
    "tags": ["karras", "unconditional"],
    "beta1": 1e-5,
    "betaT": 1e-2,
    "deterministic": False,
    "percent": 100,
    "epochs": 1000
}


def _load_and_merge_config(args) -> None:
    cfg_path = _CONFIG_DIR / f"{args.dataname}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No imagentime config for '{args.dataname}'. Expected: {cfg_path}"
        )
    config = OmegaConf.to_object(OmegaConf.load(cfg_path))
    # imagentime defaults fill in anything not already on args
    for k, v in _IMAGENTIME_DEFAULTS.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
    # dataset config always wins — project-level get_args() defaults (e.g. seq_len=24)
    # must not override the per-dataset YAML values
    for k, v in config.items():
        setattr(args, k, v)
    args.dataset = args.dataname  # imagentime internals use args.dataset
    args.epochs = 1000


def main(args) -> None:
    _load_and_merge_config(args)

    (ImagenTime, DiffusionProcess, evaluate_model_uncond,
     PrintLogger, save_checkpoint, restore_checkpoint,
     print_model_params, gen_dataloader) = _import_imagentime_internals()

    if torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}" if args.gpu is not None else "cuda"
    else:
        args.device = "cpu"

    ckpt_dir = Path(_HERE) / "checkpoints" / args.dataname
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(ckpt_dir / "model.pt")
    args.log_dir = ckpt_path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(f"[imagentime] device={args.device}  dataset={args.dataset}  ckpt={ckpt_path}")

    with PrintLogger() as logger:
        train_loader, test_loader = gen_dataloader(args)
        logging.info(f"{args.dataset} dataset ready.")

        model = ImagenTime(args=args, device=args.device).to(args.device)
        if args.use_stft:
            model.init_stft_embedder(train_loader)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        state = {"model": model, "epoch": 0}
        init_epoch = 0

        if getattr(args, "resume", False) and Path(ckpt_path).exists():
            ema_model = model.model_ema if args.ema else None
            state = restore_checkpoint(ckpt_path, state, device=args.device, ema_model=ema_model)
            init_epoch = state["epoch"]
            logging.info(f"Resumed from epoch {init_epoch}.")

        print_model_params(logger, model)

        best_score = float("inf")
        train_start = time.time()
        epoch_times = []

        for epoch in range(init_epoch, args.epochs):
            t0 = time.time()
            model.train()
            model.epoch = epoch
            logger.log_name_params("train/epoch", epoch)

            epoch_log_accum = {}
            for data in train_loader:
                x_ts = data[0].to(args.device)
                x_img = model.ts_to_img(x_ts)
                optimizer.zero_grad()
                loss = model.loss_fn(x_img)
                if len(loss) == 2:
                    loss, to_log = loss
                    for key, value in to_log.items():
                        epoch_log_accum.setdefault(key, []).append(value)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.on_train_batch_end()

            for key, values in epoch_log_accum.items():
                logger.log(f"train/{key}", sum(values) / len(values), epoch)

            if epoch % args.logging_iter == 0:
                gen_sig, real_sig = [], []
                model.eval()
                with torch.no_grad():
                    with model.ema_scope():
                        process = DiffusionProcess(
                            args,
                            model.net,
                            (args.input_channels, args.img_resolution, args.img_resolution),
                        )
                        for data in test_loader:
                            x_img_s = process.sampling(sampling_number=data[0].shape[0])
                            x_ts_s = model.img_to_ts(x_img_s)
                            if args.dataset == "temperature_rain":
                                x_ts_s = torch.clamp(x_ts_s, 0, 1)
                            gen_sig.append(x_ts_s.detach().cpu().numpy())
                            real_sig.append(data[0].detach().cpu().numpy())

                gen_sig = np.vstack(gen_sig)
                real_sig = np.vstack(real_sig)
                scores = evaluate_model_uncond(real_sig, gen_sig, args)
                for key, value in scores.items():
                    logger.log(f"test/{key}", value, epoch)

                curr_score = (
                    scores["marginal_score_mean"]
                    if "marginal_score_mean" in scores
                    else scores["disc_mean"]
                )
                if curr_score < best_score:
                    best_score = curr_score
                    ema_model = model.model_ema if args.ema else None
                    save_checkpoint(ckpt_path, state, epoch, ema_model)
                    logging.info(f"  -> checkpoint saved (epoch {epoch}, score {curr_score:.4f})")

            epoch_times.append(time.time() - t0)

        total_secs = time.time() - train_start
        avg_epoch = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
        logging.info(
            f"Training complete. "
            f"Total: {timedelta(seconds=int(total_secs))}  "
            f"Avg/epoch: {avg_epoch:.2f}s  "
            f"Best score: {best_score:.4f}"
        )

        runtime_path = ckpt_dir / "runtime.txt"
        with open(runtime_path, "w") as f:
            f.write(f"dataset:        {args.dataset}\n")
            f.write(f"device:         {args.device}\n")
            f.write(f"epochs:         {args.epochs}\n")
            f.write(f"total_time_s:   {total_secs:.2f}\n")
            f.write(f"total_time:     {timedelta(seconds=int(total_secs))}\n")
            f.write(f"avg_epoch_s:    {avg_epoch:.2f}\n")
            f.write(f"min_epoch_s:    {min(epoch_times):.2f}\n")
            f.write(f"max_epoch_s:    {max(epoch_times):.2f}\n")
            f.write(f"best_score:     {best_score:.4f}\n")
        logging.info(f"Runtime saved to {runtime_path}")
