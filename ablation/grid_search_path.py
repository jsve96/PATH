#!/usr/bin/env python3
"""
Grid search for the PATH model on kdd_cup.

Sweeps:
  K            : [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
  h            : [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
  kernel_type  : [0=quartic, 1=gaussian, 2=epanechnikov, 3=triangular]

Results are written incrementally to JSON so the run can be interrupted
and resumed at any time.

Usage:
  python grid_search_path.py                  # full run (GRU + TCN + LR + RF)
  python grid_search_path.py --fast           # GRU only (much faster)
  python grid_search_path.py --gpu 0          # use GPU for discriminator
  python grid_search_path.py --max-samples 1000  # cap synthetic/real samples
"""

import sys
import os
import json
import argparse
import itertools
import time
import traceback
from pathlib import Path

import numpy as np
import torch

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.data_utils import load_data
from models.path.model import simulate_path
from eval_utils.eval_disc_utils import (
    neural_discriminative_score_torch,
    extended_discriminative_scores,
    aggregate_discriminative_scores,
)

# ── grid definition ───────────────────────────────────────────────────────────
DATASET     = "kdd_cup"
K_VALUES    = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
H_VALUES    = [0.025]#[0.01, 0.05,0.25,0.1,0.5]#, 0.1, 0.25, 0.5, 1.0]
KERNEL_TYPES = [0, 1, 2, 3]
KERNEL_NAMES = {0: "quartic", 1: "gaussian", 2: "epanechnikov", 3: "triangular"}

RESULTS_PATH = Path("results/path_grid_kdd_cup.json")


# ── helpers ───────────────────────────────────────────────────────────────────

def _combo_key(K, h, kernel_type):
    return f"K={K}_h={h}_kernel={KERNEL_NAMES[kernel_type]}"


def _load_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def _save_results(results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def _eval_fast(real, synth, device, seed):
    """GRU only — quick proxy score."""
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
        "gru": scores,
        "aggregate": {
            "DS_acc_GRU_TCN":  scores["discriminative_score_acc"],
            "DS_auc_GRU_TCN":  scores["discriminative_score_auroc"],
            "DS_acc_LR_RF":    float("nan"),
            "DS_auc_LR_RF":    float("nan"),
        },
    }


def _eval_full(real, synth, device, seed):
    """GRU + TCN + logistic + RF."""
    scores = extended_discriminative_scores(
        X_data=real,
        X_sbts=synth,
        iterations=2000,
        batch_size=128,
        device=device,
        seed=seed,
    )
    agg = aggregate_discriminative_scores(scores)
    return {
        "gru":                  scores["gru"],
        "tcn":                  scores["tcn"],
        "summary_logistic":     scores["summary_logistic"],
        "summary_random_forest":scores["summary_random_forest"],
        "aggregate":            agg,
    }


# ── main grid search ──────────────────────────────────────────────────────────

def run_grid_search(args):
    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    print(f"[PATH grid] device={device}  fast={args.fast}  dataset={DATASET}")

    data      = load_data(DATASET)
    test_data = load_data(DATASET, train=False)
    M, N, d   = data.shape
    M_simu    = test_data.shape[0]

    if args.max_samples and args.max_samples < M_simu:
        M_simu = args.max_samples
        print(f"[PATH grid] capping M_simu at {M_simu}")

    real_eval = data[:M_simu] if args.max_samples else data

    print(f"[PATH grid] data shape: {data.shape}  M_simu={M_simu}")

    results = _load_results()

    combos = list(itertools.product(K_VALUES, H_VALUES, KERNEL_TYPES))
    total  = len(combos)
    done   = sum(1 for K, h, kt in combos if _combo_key(K, h, kt) in results)
    print(f"[PATH grid] {total} combos total — {done} already cached, "
          f"{total - done} to run\n")

    eval_fn = _eval_fast if args.fast else _eval_full

    for i, (K, h, kernel_type) in enumerate(combos):
        key = _combo_key(K, h, kernel_type)

        if key in results:
            continue

        kernel_name = KERNEL_NAMES[kernel_type]
        print(f"[{i+1:>3}/{total}] K={K:<4} h={h:<5} kernel={kernel_name:<12}", end=" ", flush=True)

        entry = {"K": K, "h": h, "kernel_type": kernel_type, "kernel_name": kernel_name}

        try:
            t0 = time.perf_counter()
            syn_samples, _, _ = simulate_path(
                N=N, M=M, d=d, K=K, X=data, h=h, M_simu=M_simu, kernel_type=kernel_type
            )
            entry["sim_time"] = round(time.perf_counter() - t0, 2)

            t0 = time.perf_counter()
            eval_out = eval_fn(real_eval, syn_samples, device=device, seed=42)
            entry["eval_time"] = round(time.perf_counter() - t0, 2)
            entry.update(eval_out)

            agg = entry["aggregate"]
            print(
                f"DS_acc={agg['DS_acc_GRU_TCN']:.4f}  "
                f"DS_auc={agg['DS_auc_GRU_TCN']:.4f}  "
                f"sim={entry['sim_time']:.1f}s  eval={entry['eval_time']:.1f}s"
            )

        except Exception:
            msg = traceback.format_exc().strip().splitlines()[-1]
            entry["error"] = msg
            print(f"ERROR: {msg}")

        results[key] = entry
        _save_results(results)

    print(f"\n[PATH grid] Done. Results: {RESULTS_PATH}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PATH grid search on kdd_cup")
    parser.add_argument("--gpu",         type=int, default=-1,
                        help="GPU index (-1 = CPU)")
    parser.add_argument("--fast",        action="store_true",
                        help="Use GRU discriminator only (faster)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap number of synthetic/real samples for evaluation")
    args = parser.parse_args()
    run_grid_search(args)
