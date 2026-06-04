import os

# Helps CUDA reproducibility for operations that depend on cuBLAS.
# This must be set before importing torch.
#os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from pathlib import Path
from collections import defaultdict
import random

import numpy as np
import pandas as pd
import torch

from eval_utils.eval_pred_utils import (
    multihorizon_tstr_trtr_score_torch,
    acf_error,
    cross_correlation_error,
)


def set_global_seed(seed, deterministic=True):
    """Seed Python, NumPy, and PyTorch for repeatable runs."""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch versions do not support warn_only.
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        except Exception:
            pass


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


def parse_int_list(value):
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def parse_seeds(seed_string):
    return [int(seed.strip()) for seed in seed_string.split(",") if seed.strip()]


def scaler_arg_to_value(scaler):
    if scaler == "none":
        return None
    return scaler


def run_predictive_pipeline(
    X_data,
    X_sbts,
    horizons=(1, 3, 5, 10),
    predictive_iterations=2000,
    batch_size=128,
    test_ratio=0.2,
    max_acf_lag=20,
    max_cross_corr_lag=0,
    scaler="standardize",
    device=torch.device("cpu"),
    seed=None,
    verbose=False,
    metric="mae",
):
    """
    Run TSTR/TRTR predictive evaluation plus ACF and cross-correlation errors.

    Returns
    -------
    dict
        TSTR/TRTR multi-horizon predictive score, ACF error, and cross-correlation error.
    """
    pred_res = multihorizon_tstr_trtr_score_torch(
        original_data=X_data,
        synthetic_data=X_sbts,
        horizons=horizons,
        iterations=predictive_iterations,
        scaler=scaler,
        batch_size=batch_size,
        test_ratio=test_ratio,
        device=device,
        seed=seed,
        verbose=verbose,
        return_per_feature=True,
        metric=metric,
    )

    acf_res = acf_error(
        original_data=X_data,
        synthetic_data=X_sbts,
        max_lag=max_acf_lag,
        return_details=False,
    )

    corr_res = cross_correlation_error(
        original_data=X_data,
        synthetic_data=X_sbts,
        max_lag=max_cross_corr_lag,
        exclude_diagonal=True,
        return_details=False,
    )

    return {
        "predictive": pred_res,
        "acf_error": acf_res,
        "cross_correlation_error": corr_res,
    }


def _flatten_predictive_block(row, prefix, block, include_per_feature=False):
    """Flatten a TSTR or TRTR predictive block."""
    row[f"{prefix}_mean_mae"] = block["mean_mae"]

    for horizon, value in block["mae_by_horizon"].items():
        row[f"{prefix}_mae_h{horizon}"] = value

    if include_per_feature and "mae_per_horizon_feature" in block:
        for horizon, arr in block["mae_per_horizon_feature"].items():
            arr = np.asarray(arr).reshape(-1)
            for feature_idx, value in enumerate(arr):
                row[f"{prefix}_mae_h{horizon}_feature{feature_idx}"] = float(value)


def flatten_predictive_results(
    results,
    seed,
    include_per_feature=False,
    include_acf_details=True,
    include_cross_corr_details=True,
):
    """
    Flatten nested metric dictionaries into scalar columns for pandas.describe().
    """
    row = {"seed": seed}

    pred = results["predictive"]
    tstr = pred["TSTR"]
    trtr = pred["TRTR"]

    _flatten_predictive_block(row, "tstr", tstr, include_per_feature=include_per_feature)
    _flatten_predictive_block(row, "trtr", trtr, include_per_feature=include_per_feature)

    row["pred_gap_mean_mae"] = pred["gap_mean_mae"]
    row["pred_ratio_mean_mae"] = pred["ratio_mean_mae"]

    for horizon, value in pred["gap_by_horizon"].items():
        row[f"pred_gap_mae_h{horizon}"] = value

    for horizon, value in pred["ratio_by_horizon"].items():
        row[f"pred_ratio_mae_h{horizon}"] = value

    # Persistence baseline + skill scores
    if "PERSIST" in pred:
        _flatten_predictive_block(row, "persist", pred["PERSIST"])
        if "PERSIST_TSTR" in pred:
            _flatten_predictive_block(row, "persist_tstr", pred["PERSIST_TSTR"])
        row["tstr_skill_mean"] = pred["tstr_skill_mean"]
        row["trtr_skill_mean"] = pred["trtr_skill_mean"]
        for horizon, value in pred["tstr_skill_by_horizon"].items():
            row[f"tstr_skill_h{horizon}"] = value
        for horizon, value in pred["trtr_skill_by_horizon"].items():
            row[f"trtr_skill_h{horizon}"] = value

    # Backward-compatible aliases: previous pipeline reported only TSTR as pred_*.
    row["pred_mean_mae"] = tstr["mean_mae"]
    for horizon, value in tstr["mae_by_horizon"].items():
        row[f"pred_mae_h{horizon}"] = value

    acf = results["acf_error"]
    row["acf_error"] = acf["acf_error"]

    if include_acf_details:
        if "acf_error_by_lag" in acf:
            for lag_idx, value in enumerate(np.asarray(acf["acf_error_by_lag"]).reshape(-1), start=1):
                row[f"acf_error_lag{lag_idx}"] = float(value)

        if include_per_feature and "acf_error_by_feature" in acf:
            for feature_idx, value in enumerate(np.asarray(acf["acf_error_by_feature"]).reshape(-1)):
                row[f"acf_error_feature{feature_idx}"] = float(value)

    corr = results["cross_correlation_error"]
    row["cross_corr_error"] = corr["cross_corr_error"]

    if include_cross_corr_details and "cross_corr_error_by_lag" in corr:
        for lag_idx, value in enumerate(np.asarray(corr["cross_corr_error_by_lag"]).reshape(-1)):
            row[f"cross_corr_error_lag{lag_idx}"] = float(value)

    return row


def aggregate_runs(summary):
    """
    Convert list of per-seed dictionaries into:
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
        description="Run extended predictive, TRTR/TSTR, ACF, and cross-correlation metrics over multiple seeds."
    )

    parser.add_argument(
        "--dataname",
        required=True,
        help="Name to real data. Shape should be (N, T, D).",
    )

    parser.add_argument(
        "--method",
        required=True,
        help="Method to analyze. Shape should be (N, T, D).",
    )

    parser.add_argument(
        "--real-key",
        default=None,
        help="Optional key for real data if using .npz or dict-like .pt/.pth.",
    )

    parser.add_argument(
        "--syn-key",
        default=None,
        help="Optional key for synthetic data if using .npz or dict-like .pt/.pth.",
    )

    parser.add_argument(
        "--real-start",
        type=int,
        default=0,
        help="Start index on the time axis for real data. Use 1 to mimic X_data[:, 1:, :].",
    )

    parser.add_argument(
        "--syn-start",
        type=int,
        default=0,
        help="Start index on the time axis for synthetic data.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=5000,
        help="Number of training iterations for each multi-horizon predictor.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for each multi-horizon predictor.",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Held-out real test ratio used for TRTR and TSTR evaluation.",
    )

    parser.add_argument(
        "--horizons",
        type=str,
        default="1,3,5,10",
        help="Comma-separated predictive horizons, e.g. '1,3,5,10'.",
    )

    parser.add_argument(
        "--max-acf-lag",
        type=int,
        default=20,
        help="Maximum lag for ACF error.",
    )

    parser.add_argument(
        "--max-cross-corr-lag",
        type=int,
        default=0,
        help="Maximum lag for cross-correlation error. Use 0 for contemporaneous feature correlation.",
    )

    parser.add_argument(
        "--scaler",
        choices=["standardize", "normalize", "none"],
        default="none",
        help="Scaling used by the predictive score.",
    )

    parser.add_argument(
        "--metric",
        choices=["mae", "mse", "rmse", "huber"],
        default="rmse",
        help=(
            "Loss function used for both predictor training and evaluation. "
            "'mse'/'rmse' penalise large per-feature errors more than 'mae'; "
            "'huber' is a smooth compromise between the two."
        ),
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="42,0,1,2,3",
        help="Comma-separated list of seeds.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for the predictive model.",
    )

    parser.add_argument(
        "--include-per-feature",
        action="store_true",
        help="Include per-feature predictive and ACF columns in the raw/summary CSV files.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print predictor training loss every 500 iterations.",
    )


    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seeds = parse_seeds(args.seeds)
    horizons = parse_int_list(args.horizons)
    scaler = scaler_arg_to_value(args.scaler)



    #X_data = load_array(args.real, key=args.real_key)
    #X_sbts = load_array(args.syn, key=args.syn_key)

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
    print(f"Horizons: {horizons}")
    print(f"Scaler: {args.scaler}")
    print(f"Metric: {args.metric}")
    print(f"Test ratio: {args.test_ratio}")

    summary = []

    for seed in seeds:
        print(f"\nRunning seed {seed}...")
        set_global_seed(seed)

        results = run_predictive_pipeline(
            X_data=X_real,
            X_sbts=X_syn,
            horizons=horizons,
            predictive_iterations=args.iterations,
            batch_size=args.batch_size,
            test_ratio=args.test_ratio,
            max_acf_lag=args.max_acf_lag,
            max_cross_corr_lag=args.max_cross_corr_lag,
            scaler=scaler,
            device=device,
            seed=seed,
            verbose=args.verbose,
            metric=args.metric,
        )

        row = flatten_predictive_results(
            results,
            seed=seed,
            include_per_feature=args.include_per_feature,
        )
        summary.append(row)

        horizon_msg = ", ".join(
            f"TSTR@{h}={row[f'tstr_mae_h{h}']:.4f}, TRTR@{h}={row[f'trtr_mae_h{h}']:.4f}"
            for h in horizons
        )
        skill_str = (
            f"TSTR_skill={row['tstr_skill_mean']:.3f}, "
            f"TRTR_skill={row['trtr_skill_mean']:.3f}, "
            f"Persist_TRTR={row['persist_mean_mae']:.4f}"
            + (f", Persist_TSTR={row['persist_tstr_mean_mae']:.4f}" if "persist_tstr_mean_mae" in row else "")
            if "tstr_skill_mean" in row else ""
        )
        print(
            f"Seed {seed} | "
            f"TSTR={row['tstr_mean_mae']:.4f}, "
            f"TRTR={row['trtr_mean_mae']:.4f}, "
            f"Gap={row['pred_gap_mean_mae']:.4f}, "
            f"Ratio={row['pred_ratio_mean_mae']:.4f}, "
            f"{horizon_msg}"
            + (f", {skill_str}" if skill_str else "") +
            f", ACF={row['acf_error']:.4f}, "
            f"CrossCorr={row['cross_corr_error']:.4f}"
        )

    _ , desc_df = aggregate_runs(summary)


    save_dir = f"results/pred/{args.dataname}"
    os.makedirs(save_dir, exist_ok=True)

    desc_df.to_csv(f"{save_dir}/{args.method}.csv")

    print("\nSummary statistics:")
    print(desc_df)
    
    #print(f"\nSaved per-seed results to: {args.raw_out}")
    print(f"Saved summary statistics to: {save_dir}/{args.method}.csv")

if __name__ == "__main__":
    main()
