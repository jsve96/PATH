"""
Privacy evaluation script for time series synthesis.

Measures two complementary memorisation signals:

1. Nearest-Neighbour Distance (NND)
   Compares d(synthetic → training) vs. d(real-test → training).
   If synthetic trajectories systematically sit closer to training data than
   held-out real trajectories do, this indicates memorisation.

   Key output: nn_ratio_mean  (ideal ≈ 1; < 1 is suspicious)
               pct_gen_below_test_median (ideal ≈ 0.5; >> 0.5 flags concern)
               ks_stat / ks_pvalue

2. Subsequence Copying
   For each generated trajectory, finds the longest contiguous block that
   near-matches any training trajectory (all temporal offsets).
   Lower L_max / T means less direct copying.

   Key output: subseq_lmax_norm_mean  (ideal ≈ 0)
               subseq_lmax_frac       (fraction with any copy detected)

Usage
-----
python run_eval_privacy.py --dataname stocks --method path

# Skip the slower subsequence check
python run_eval_privacy.py --dataname stocks --method tsdiff --no-subseq

# Use a custom tau or limit comparison size for speed
python run_eval_privacy.py --dataname stocks --method path \\
    --subseq-tau 0.05 --subseq-n-train 100 --subseq-n-syn 200

Results are saved to results/privacy/{dataname}/{method}.csv
"""

import os
import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eval_utils.eval_privacy_utils import (
    privacy_nn_metrics,
    privacy_mia_metrics,
    privacy_subseq_metrics,
    auto_tau,
)


# ── I/O helpers (mirrored from other eval scripts) ────────────────────────────

def load_array(path, key=None):
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        if key is None:
            if len(data.files) != 1:
                raise ValueError(
                    f"{path} contains multiple arrays: {data.files}. "
                    "Specify --real-key or --syn-key."
                )
            key = data.files[0]
        arr = data[key]
    elif path.suffix in (".pt", ".pth"):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy()
        elif isinstance(obj, dict):
            if key is None:
                if len(obj) != 1:
                    raise ValueError(
                        f"{path} has multiple keys: {list(obj.keys())}. "
                        "Specify --real-key or --syn-key."
                    )
                key = list(obj.keys())[0]
            v = obj[key]
            arr = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
        else:
            arr = np.asarray(obj)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    return np.asarray(arr, dtype=np.float32)


def ensure_3d(x, name):
    if x.ndim == 2:
        x = x[:, :, None]
    if x.ndim != 3:
        raise ValueError(f"{name} must be (N, T, D). Got {x.shape}.")
    return x


def parse_seeds(s):
    return [int(v.strip()) for v in s.split(",") if v.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def aggregate_runs(summary):
    merged = defaultdict(list)
    for d in summary:
        for k, v in d.items():
            merged[k].append(v)
    df = pd.DataFrame(merged)
    return df, df.describe()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Privacy evaluation: nearest-neighbour distances and subsequence copying.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data (same conventions as run_eval_disc / run_eval_pred)
    parser.add_argument("--dataname",   required=True, help="Dataset name.")
    parser.add_argument("--method",     required=True, help="Method name.")
    parser.add_argument("--real-key",   default=None,  help="Key for real data in .npz/.pt.")
    parser.add_argument("--syn-key",    default=None,  help="Key for synthetic data in .npz/.pt.")
    parser.add_argument("--real-start", type=int, default=0,
                        help="Time-axis start index for real data.")
    parser.add_argument("--syn-start",  type=int, default=0,
                        help="Time-axis start index for synthetic data.")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Held-out real test split.")
    parser.add_argument("--seeds",      type=str, default="42,0,1,2,3",
                        help="Comma-separated random seeds.")

    # NND options
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable per-element normalisation of L2 distances.")
    parser.add_argument("--batch-size",   type=int, default=256,
                        help="Batch size for pairwise L2 computation.")

    # Subsequence options
    parser.add_argument("--no-subseq", action="store_true",
                        help="Skip subsequence-copying evaluation (much faster).")
    parser.add_argument("--subseq-tau", type=float, default=None,
                        help="Per-timestep L2 threshold.  Auto-computed if not set.")
    parser.add_argument("--subseq-tau-quantile", type=float, default=0.1,
                        help="Quantile of intra-training distances used for auto tau.")
    parser.add_argument("--subseq-min-len", type=int, default=2,
                        help="Minimum subsequence length to search for.")
    parser.add_argument("--subseq-n-train", type=int, default=200,
                        help="Max training trajectories compared per synthetic sample.")
    parser.add_argument("--subseq-n-syn",   type=int, default=None,
                        help="Subsample synthetic set for subsequence check (default: all).")
    parser.add_argument("--subseq-batch-size", type=int, default=256,
                        help="Synthetic samples per vectorised batch. "
                             "Memory ≈ batch × n_train × T² × 4 B. "
                             "Reduce for large T (e.g. --subseq-batch-size 32 for T=100).")
    parser.add_argument("--subseq-aligned-only", action="store_true",
                        help="Only check d=0 (same-time-alignment) offset. Much faster; "
                             "less sensitive to shifted copies.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-batch progress during subsequence computation.")

    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)

    X_real = ensure_3d(load_array(f"data/{args.dataname}/test.npy",                         key=args.real_key), "real")
    X_syn  = ensure_3d(load_array(f"synthetic/{args.dataname}/{args.method}.npy",       key=args.syn_key),  "synthetic")

    X_real = X_real[:, args.real_start:, :]
    X_syn  = X_syn[:,  args.syn_start:,  :]

    T = X_real.shape[1]

    print(f"Real data shape:      {X_real.shape}")
    print(f"Synthetic data shape: {X_syn.shape}")
    print(f"Seeds: {seeds}")
    print(f"Test ratio: {args.test_ratio}")

    summary = []

    for seed in seeds:
        print(f"\n{'─'*50}")
        print(f"Seed {seed}")
        print(f"{'─'*50}")
        set_seed(seed)
        rng = np.random.default_rng(seed)

        # Train / test split of real data (deterministic per seed)
        N = X_real.shape[0]
        idx = rng.permutation(N)
        n_test  = max(1, int(N * args.test_ratio))
        X_test  = X_real[idx[:n_test]]
        X_train = X_real[idx[n_test:]]

        # ── 1. Nearest-Neighbour Distance ─────────────────────────────────
        print("NND: computing nearest-neighbour distances...")
        nn = privacy_nn_metrics(
            X_train=X_train,
            X_test=X_test,
            X_syn=X_syn,
            normalize=not args.no_normalize,
            batch_size=args.batch_size,
        )

        print(
            f"  d_gen_mean={nn['nn_dist_gen_mean']:.4f}  "
            f"d_test_mean={nn['nn_dist_test_mean']:.4f}  "
            f"ratio={nn['nn_ratio_mean']:.4f}  "
            f"KS={nn['ks_stat']:.4f} (p={nn['ks_pvalue']:.3f})  "
            f"pct_below_median={nn['pct_gen_below_test_median']:.4f}"
        )

        row = {"seed": seed, **nn}

        # ── 2. Membership Inference Attack proxy ──────────────────────────
        print("MIA: computing nearest-neighbour distances to synthetic set...")
        mia = privacy_mia_metrics(
            X_train=X_train,
            X_test=X_test,
            X_syn=X_syn,
            normalize=not args.no_normalize,
            batch_size=args.batch_size,
        )
        print(
            f"  mia_auc={mia['mia_auc']:.4f}  "
            f"advantage={mia['mia_advantage']:.4f}  "
            f"d_member_mean={mia['d_member_mean']:.4f}  "
            f"d_nonmember_mean={mia['d_nonmember_mean']:.4f}"
        )
        row.update(mia)

        # ── 3. Subsequence Copying ────────────────────────────────────────
        if not args.no_subseq:
            tau = args.subseq_tau
            if tau is None:
                tau = auto_tau(
                    X_train,
                    quantile=args.subseq_tau_quantile,
                    n_pairs=500,
                    rng=rng,
                )
                print(f"  auto tau = {tau:.6f}  (quantile={args.subseq_tau_quantile})")

            X_syn_eval = X_syn
            if args.subseq_n_syn is not None and args.subseq_n_syn < X_syn.shape[0]:
                sub_idx = rng.choice(X_syn.shape[0], size=args.subseq_n_syn, replace=False)
                X_syn_eval = X_syn[sub_idx]

            n_train_used = min(args.subseq_n_train, X_train.shape[0])
            print(
                f"Subsequence: tau={tau:.4f}  n_syn={X_syn_eval.shape[0]}  "
                f"n_train={n_train_used}  min_len={args.subseq_min_len}"
            )

            subseq = privacy_subseq_metrics(
                X_syn=X_syn_eval,
                X_train=X_train,
                tau=tau,
                min_len=args.subseq_min_len,
                n_train_max=args.subseq_n_train,
                batch_size=args.subseq_batch_size,
                aligned_only=args.subseq_aligned_only,
                rng=rng,
                verbose=args.verbose,
            )

            print(
                f"  lmax_mean={subseq['subseq_lmax_mean']:.2f}/{T}  "
                f"(norm={subseq['subseq_lmax_norm_mean']:.3f})  "
                f"lmax_max={subseq['subseq_lmax_max']}  "
                f"frac_any={subseq['subseq_lmax_frac']:.4f}"
            )
            row.update(subseq)

        summary.append(row)

    # ── Save results ──────────────────────────────────────────────────────────
    _, desc_df = aggregate_runs(summary)

    save_dir = f"results/privacy/{args.dataname}"
    os.makedirs(save_dir, exist_ok=True)
    out_path = f"{save_dir}/{args.method}.csv"
    desc_df.to_csv(out_path)

    print(f"\n{'═'*50}")
    print("Summary statistics:")
    print(desc_df[['nn_dist_gen_mean','nn_dist_test_mean','nn_ratio_mean','mia_auc','mia_advantage']])
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
