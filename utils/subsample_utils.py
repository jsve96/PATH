import json
import shutil
from pathlib import Path

import numpy as np


def fraction_tag(r: float) -> str:
    """Convert a fraction to a zero-padded string tag, e.g. 0.10 → 'r010'."""
    return f"r{int(round(r * 100)):03d}"


def subsample_and_save(
    dataname: str,
    fraction: float,
    seed: int = 42,
    data_root: str = "data",
) -> str:
    """Subsample the training set of *dataname* and write a new dataset folder.

    The test set is copied unchanged so all three evaluation regimes are
    assessed against the same held-out split used in the main experiments.

    Parameters
    ----------
    dataname : str
        Source dataset name, e.g. ``"stocks"``.
    fraction : float
        Fraction of training samples to retain, e.g. ``0.10``.
    seed : int
        Random seed for the subsample draw.
    data_root : str
        Root folder containing per-dataset sub-directories.

    Returns
    -------
    str
        The new dataset name, e.g. ``"stocks_r010"``.

    Raises
    ------
    ValueError
        If *fraction* is not strictly in (0, 1).
    FileNotFoundError
        If train.npy or test.npy cannot be found under *src*.
    """
    if not (0.0 < fraction < 1.0):
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")

    src = Path(data_root) / dataname
    tag = fraction_tag(fraction)
    new_name = f"{dataname}_{tag}"
    dst = Path(data_root) / new_name
    dst.mkdir(parents=True, exist_ok=True)

    train_src = src / "train.npy"
    if not train_src.exists():
        raise FileNotFoundError(f"Training data not found: {train_src}")

    train_data = np.load(train_src)  # (N, T, D)
    n_total = len(train_data)
    n_subset = max(1, round(fraction * n_total))

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_subset, replace=False)
    idx.sort()  # preserve relative order for reproducibility auditing
    subset = train_data[idx]

    np.save(dst / "train.npy", subset)

    test_src = src / "test.npy"
    if not test_src.exists():
        raise FileNotFoundError(f"Test data not found: {test_src}")
    shutil.copy2(test_src, dst / "test.npy")

    meta = {
        "source_dataname": dataname,
        "fraction": fraction,
        "tag": tag,
        "seed": seed,
        "n_total": n_total,
        "n_subset": n_subset,
        "indices": idx.tolist(),
    }
    with open(dst / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(
        f"[subsample] {dataname} -> {new_name}: "
        f"{n_subset}/{n_total} samples  (seed={seed})"
    )
    return new_name



if __name__ == '__main__':
    datasets = ["stocks", "energy", "air"]
    fractions = [0.10, 0.20]

    for ds in datasets:
        for r in fractions:
            subsample_and_save(ds, r, seed=42)
