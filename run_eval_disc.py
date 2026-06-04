import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

import os

from eval_utils.eval_disc_utils import *


def load_array(path, key=None):
    """
    Load array from .npy, .npz, .pt, or .pth file.

    Parameters
    ----------
    path : str
        Path to input file.

    key : str | None
        Key used for .npz or dict-like .pt/.pth files.

    Returns
    -------
    np.ndarray
    """
    path = Path(path)

    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=False)

    elif path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)

        if key is None:
            if len(data.files) != 1:
                raise ValueError(
                    f"{path} contains multiple arrays: {data.files}. "
                    "Please specify the key using --real-key or --syn-key."
                )
            key = data.files[0]

        arr = data[key]

    elif path.suffix in [".pt", ".pth"]:
        obj = torch.load(path, map_location="cpu")

        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy()

        elif isinstance(obj, dict):
            if key is None:
                if len(obj.keys()) != 1:
                    raise ValueError(
                        f"{path} contains multiple keys: {list(obj.keys())}. "
                        "Please specify the key using --real-key or --syn-key."
                    )
                key = list(obj.keys())[0]

            value = obj[key]
            if isinstance(value, torch.Tensor):
                arr = value.detach().cpu().numpy()
            else:
                arr = np.asarray(value)

        else:
            arr = np.asarray(obj)

    else:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            "Supported formats are .npy, .npz, .pt, and .pth."
        )

    return np.asarray(arr, dtype=np.float32)


def ensure_3d(x, name):
    """
    Ensure data has shape (N, T, D).
    If data has shape (N, T), convert to (N, T, 1).
    """
    if x.ndim == 2:
        x = x[:, :, None]

    if x.ndim != 3:
        raise ValueError(f"{name} must have shape (N, T, D). Got {x.shape}.")

    return x


def parse_seeds(seed_string):
    return [int(seed.strip()) for seed in seed_string.split(",") if seed.strip()]


def aggregate_runs(summary):
    """
    Convert list of aggregate dictionaries into:
        raw dataframe
        describe dataframe
    """
    merged = defaultdict(list)

    for d in summary:
        for key, value in d.items():
            merged[key].append(value)

    df = pd.DataFrame(merged)
    desc = df.describe()

    return df, desc


def main():
    parser = argparse.ArgumentParser(
        description="Run extended discriminative scores over multiple seeds."
    )
    parser.add_argument(
        "--dataname",
        required=True,
        help="Name of dataset. Shape should be (N, T, D)."
    )

    parser.add_argument(
        "--method",
        required=True,
        help="Method for syn. Shape should be (N, T, D)."
    )

    parser.add_argument(
        "--real-key",
        default=None,
        help="Optional key for real data if using .npz or dict-like .pt/.pth."
    )

    parser.add_argument(
        "--syn-key",
        default=None,
        help="Optional key for synthetic data if using .npz or dict-like .pt/.pth."
    )

    parser.add_argument(
        "--real-start",
        type=int,
        default=0,
        help="Start index on the time axis for real data. Use 1 to mimic X_data[:, 1:, :]."
    )

    parser.add_argument(
        "--syn-start",
        type=int,
        default=0,
        help="Start index on the time axis for synthetic data."
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=5000,
        help="Number of training iterations for GRU and TCN discriminators."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for neural discriminators."
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Held-out test ratio."
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="42,0,1,2,3",
        help="Comma-separated list of seeds."
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for neural discriminators."
    )

    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable standardization inside extended_discriminative_scores."
    )



    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seeds = parse_seeds(args.seeds)

    X_real = load_array(f"data/{args.dataname}/test.npy", key=args.real_key)
    X_syn = load_array(f"synthetic/{args.dataname}/{args.method}.npy", key=args.syn_key)

    X_real = ensure_3d(X_real, "real")
    X_syn = ensure_3d(X_syn, "synthetic")

    X_real = X_real[:, args.real_start:, :]
    X_syn = X_syn[:, args.syn_start:, :]

    print(f"Real data shape:      {X_real.shape}")
    print(f"Synthetic data shape: {X_syn.shape}")
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")

    summary = []

    for seed in seeds:
        print(f"\nRunning seed {seed}...")

        disc_results = extended_discriminative_scores(
            X_data=X_real,
            X_syn=X_syn,
            iterations=args.iterations,
            batch_size=args.batch_size,
            test_ratio=args.test_ratio,
            standardize=not args.no_standardize,
            device=device,
            seed=seed,
        )

        aggregate = aggregate_discriminative_scores(disc_results)
        aggregate["seed"] = seed

        summary.append(aggregate)

        print(
            f"Seed {seed} | "
            f"DS_acc_GRU_TCN={aggregate['DS_acc_GRU_TCN']:.4f}, "
            f"DS_auc_GRU_TCN={aggregate['DS_auc_GRU_TCN']:.4f}, "
            f"DS_acc_LR_RF={aggregate['DS_acc_LR_RF']:.4f}, "
            f"DS_auc_LR_RF={aggregate['DS_auc_LR_RF']:.4f}"
        )

    _, desc_df = aggregate_runs(summary)

    save_dir = f"results/disc/{args.dataname}"
    os.makedirs(save_dir,exist_ok=True)

    desc_df.to_csv(f"{save_dir}/{args.method}.csv")

    print("\nSummary statistics:")
    print(desc_df)
    
    #print(f"\nSaved per-seed results to: {args.raw_out}")
    print(f"Saved summary statistics to: {save_dir}/{args.method}.csv")


if __name__ == "__main__":

    main()