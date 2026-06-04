#!/usr/bin/env python3
"""
Scalability experiment for PATH.

Varies one factor at a time (N_train, T, d) while fixing the other two at
baseline values.  Measures peak RAM (MB), simulation time, evaluation time,
and discriminative scores using synthetic sine data.

Usage:
    python ablation/scalability_path.py                  # full eval (GRU+TCN+LR+RF)
    python ablation/scalability_path.py --fast           # GRU only (much faster)
    python ablation/scalability_path.py --gpu 0          # use GPU for discriminator
    python ablation/scalability_path.py --seeds 0 1 2
    python ablation/scalability_path.py --out results/path_scalability_custom.json
"""

import sys
import os
import json
import time
import threading
import argparse
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.generate_data import sine_data_generation
from models.path.model import simulate_path
from eval_utils.eval_disc_utils import (
    neural_discriminative_score_torch,
    extended_discriminative_scores,
    aggregate_discriminative_scores,
)

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    import tracemalloc

# ── fixed PATH hyperparameters (best from grid search) ───────────────────────
KERNEL = "quartic"
K_PATH = 50
H_PATH = 0.05

# ── OFAT baselines ────────────────────────────────────────────────────────────
BASELINE_N = 1000
BASELINE_T = 24
BASELINE_D = 6

# ── ablation grids ────────────────────────────────────────────────────────────
N_VALUES = [100, 500, 1000, 2500, 5000, 10000, 25000]
T_VALUES = [24, 50, 100, 200, 500, 1000, 5000, 10000]
D_VALUES = [1, 3, 6, 12, 24, 48, 96]

DEFAULT_SEEDS = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
RESULTS_PATH  = Path("results/path_scalability.json")


# ── memory helpers ────────────────────────────────────────────────────────────

def _peak_rss_mb_psutil(fn, *args, **kwargs):
    """Run fn and return (result, peak_rss_mb) via a 5 ms polling thread."""
    proc = psutil.Process()
    peak = [proc.memory_info().rss]
    stop = threading.Event()

    def _monitor():
        while not stop.is_set():
            try:
                peak[0] = max(peak[0], proc.memory_info().rss)
            except Exception:
                pass
            stop.wait(0.005)

    t = threading.Thread(target=_monitor, daemon=True)
    t.start()
    result = fn(*args, **kwargs)
    stop.set()
    t.join()
    return result, peak[0] / 1024 ** 2


def _peak_mem_tracemalloc(fn, *args, **kwargs):
    """Fallback: tracemalloc (Python objects only — underestimates numpy)."""
    tracemalloc.start()
    result = fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1024 ** 2


def _run_with_mem(fn, *args, **kwargs):
    if _HAS_PSUTIL:
        return _peak_rss_mb_psutil(fn, *args, **kwargs)
    return _peak_mem_tracemalloc(fn, *args, **kwargs)


# ── resume helpers ────────────────────────────────────────────────────────────

def _key(axis: str, value: int, seed: int) -> str:
    return f"axis={axis}_val={value}_seed={seed}"


def _load(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


# ── numba warm-up ─────────────────────────────────────────────────────────────

def _warmup() -> None:
    """Trigger numba JIT compilation so it is excluded from all timings."""
    print("[scalability] Numba warm-up ...", end=" ", flush=True)
    data = sine_data_generation(20, 10, 2).astype(np.float32)
    simulate_path(N=10, M=20, d=2, K=5, X=data, h=H_PATH,
                  M_simu=10, kernel_type=KERNEL)
    print("done.\n")


# ── discriminator helpers ─────────────────────────────────────────────────────

def _eval_fast(real, synth, device, seed):
    scores = neural_discriminative_score_torch(
        original_data=real,
        synthetic_data=synth,
        architecture="gru",
        iterations=2000,
        batch_size=128,
        device=device,
        seed=seed,
    )
    return {
        "DS_acc_GRU_TCN": scores["discriminative_score_acc"],
        "DS_auc_GRU_TCN": scores["discriminative_score_auroc"],
        "DS_acc_LR_RF":   float("nan"),
        "DS_auc_LR_RF":   float("nan"),
    }


def _eval_full(real, synth, device, seed):
    scores = extended_discriminative_scores(
        X_data=real,
        X_sbts=synth,
        iterations=2000,
        batch_size=128,
        device=device,
        seed=seed,
    )
    return aggregate_discriminative_scores(scores)


# ── single measurement ────────────────────────────────────────────────────────

def measure(N_train: int, T: int, d: int, seed: int,
            device: torch.device, fast: bool) -> dict:
    np.random.seed(seed)
    data   = sine_data_generation(N_train, T, d).astype(np.float32)
    k_eff  = min(K_PATH, N_train - 1)
    M_simu = N_train

    # ── simulation ──
    def _sim():
        return simulate_path(
            N=T, M=N_train, d=d, K=k_eff, X=data,
            h=H_PATH, M_simu=M_simu, kernel_type=KERNEL,
        )

    t0 = time.perf_counter()
    (syn, _, _), peak_mb = _run_with_mem(_sim)
    sim_time = time.perf_counter() - t0

    # ── discriminative evaluation ──
    eval_fn = _eval_fast if fast else _eval_full
    t0 = time.perf_counter()
    agg = eval_fn(data, syn, device=device, seed=seed)
    eval_time = time.perf_counter() - t0

    return {
        "axis_N":        N_train,
        "axis_T":        T,
        "axis_d":        d,
        "seed":          seed,
        "K_eff":         k_eff,
        "M_simu":        M_simu,
        "sim_time_s":    round(sim_time, 4),
        "eval_time_s":   round(eval_time, 4),
        "peak_mem_mb":   round(peak_mb, 2),
        "throughput":    round(M_simu / sim_time, 2),
        "mem_backend":   "psutil" if _HAS_PSUTIL else "tracemalloc",
        "aggregate":     agg,
    }


# ── main grid loop ────────────────────────────────────────────────────────────

def run(args) -> None:
    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    print(f"[scalability] device={device}  fast={args.fast}\n")

    results = _load(args.out)

    if args.rerun_axis:
        dropped = [k for k in list(results) if any(
            k.startswith(f"axis={ax}_") for ax in args.rerun_axis
        )]
        for k in dropped:
            del results[k]
        if dropped:
            print(f"[scalability] Cleared {len(dropped)} cached entries for "
                  f"axis={args.rerun_axis} — will re-run.\n")
        _save(results, args.out)

    _warmup()

    configs = (
        [("N", v, s) for v in N_VALUES for s in args.seeds] +
        [("T", v, s) for v in T_VALUES for s in args.seeds] +
        [("d", v, s) for v in D_VALUES for s in args.seeds]
    )

    total = len(configs)
    done  = sum(1 for ax, v, s in configs if _key(ax, v, s) in results)
    print(f"[scalability] {total} configs — {done} cached, {total - done} to run\n")

    for i, (axis, val, seed) in enumerate(configs):
        key = _key(axis, val, seed)
        if key in results:
            continue

        N = val if axis == "N" else BASELINE_N
        T = val if axis == "T" else BASELINE_T
        d = val if axis == "d" else BASELINE_D

        print(f"[{i+1:>3}/{total}]  axis={axis}  val={val:<6}  seed={seed}", end="  ", flush=True)

        try:
            entry = measure(N, T, d, seed, device=device, fast=args.fast)
            agg = entry["aggregate"]
            print(
                f"sim={entry['sim_time_s']:.3f}s  "
                f"eval={entry['eval_time_s']:.1f}s  "
                f"mem={entry['peak_mem_mb']:.1f} MB  "
                f"DS_acc={agg['DS_acc_GRU_TCN']:.4f}"
            )
        except Exception:
            msg = traceback.format_exc().strip().splitlines()[-1]
            entry = {"axis_N": N, "axis_T": T, "axis_d": d,
                     "seed": seed, "error": msg}
            print(f"ERROR: {msg}")

        results[key] = entry
        _save(results, args.out)

    print(f"\n[scalability] Done → {args.out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PATH scalability experiment")
    parser.add_argument("--out",   type=Path, default=RESULTS_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--fast",        action="store_true",
                        help="GRU discriminator only (faster)")
    parser.add_argument("--gpu",         type=int, default=-1,
                        help="GPU index for discriminator (-1 = CPU)")
    parser.add_argument("--rerun-axis",  type=str, nargs="+", default=[],
                        metavar="AXIS",
                        help="Clear cached results for these axes and re-run "
                             "(e.g. --rerun-axis d  or  --rerun-axis N T)")
    args = parser.parse_args()
    run(args)
