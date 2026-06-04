"""
Shared-irregular-grid recovery experiment for PATH.

Data format
-----------
X_full has shape (M, T, d), e.g. (2928, 24, 6).
The original regular grid is taken to be 0, 1, ..., T-1.

Experiment
----------
1. Randomly drop the same interior time indices for all trajectories.
2. Run PATH on the retained irregular grid.
3. Use Markov-conditioned Brownian-bridge interpolation to recover samples
   on the original regular grid.
4. Return syn_full with shape (M_simu, T, d), which can be evaluated against
   real full-grid data with shape (M_eval, T, d).

Expected generator import
-------------------------
This file expects simulate_path_with_markov_sigma to be available either from
    .model
or from
    path_generator_knn_markov_sigma
Adjust the import below to your project structure if needed.
"""

import argparse
import os
import numpy as np

try:
    # Use this when the function is inside your package's model.py
    from models.path_ada.model_irr import simulate_path_with_markov_sigma
except ImportError:
    # Use this when path_generator_knn_markov_sigma.py is in the same folder
    from path_generator_knn_markov_sigma import simulate_path_with_markov_sigma


def make_shared_irregular_grid(T, drop_frac=0.3, n_drop=None, seed=0, keep_endpoints=True):
    """
    Randomly drop the same time indices for all trajectories.

    Parameters
    ----------
    T : int
        Number of original time observations.
    drop_frac : float
        Fraction of eligible indices to drop. Ignored if n_drop is given.
    n_drop : int or None
        Absolute number of eligible indices to drop.
    seed : int
        Random seed.
    keep_endpoints : bool
        If True, never drop 0 or T-1. Recommended because recovery then only
        requires interpolation, not extrapolation.

    Returns
    -------
    keep_idx : np.ndarray, shape (T - n_drop,)
        Sorted retained time indices.
    drop_idx : np.ndarray, shape (n_drop,)
        Sorted dropped time indices.
    full_times : np.ndarray, shape (T,)
        Original regular grid 0, ..., T-1.
    irregular_times : np.ndarray, shape (len(keep_idx),)
        Retained irregular grid.
    """
    rng = np.random.default_rng(seed)
    full_times = np.arange(T, dtype=np.float64)

    if keep_endpoints:
        eligible = np.arange(1, T - 1)
    else:
        eligible = np.arange(T)

    if n_drop is None:
        n_drop = int(np.round(drop_frac * len(eligible)))

    n_drop = int(n_drop)
    if n_drop < 0 or n_drop > len(eligible):
        raise ValueError(f"n_drop must be in [0, {len(eligible)}], got {n_drop}.")

    drop_idx = np.sort(rng.choice(eligible, size=n_drop, replace=False))

    keep_mask = np.ones(T, dtype=bool)
    keep_mask[drop_idx] = False
    keep_idx = np.where(keep_mask)[0]
    irregular_times = full_times[keep_idx]

    return keep_idx, drop_idx, full_times, irregular_times


def resolve_q(q, M):
    """
    Interpret q <= 1 as a fraction of M and q > 1 as an absolute neighbor count.
    """
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}.")

    if q <= 1:
        q_count = max(1, int(np.ceil(q * M)))
    else:
        q_count = int(q)

    return min(q_count, M)


def sample_bridge_to_target_grid(grid_path, source_times, target_times, sigma_diag_by_interval, rng):
    """
    Sample a Brownian bridge on arbitrary target times.

    This is useful when PATH generates endpoints on an irregular source grid
    and we want values on the original regular grid.

    Parameters
    ----------
    grid_path : np.ndarray, shape (N_source, d)
        Generated endpoints on the irregular grid.
    source_times : np.ndarray, shape (N_source,)
        Increasing source times.
    target_times : np.ndarray, shape (T,)
        Increasing target times. Must lie inside the source-time range.
    sigma_diag_by_interval : np.ndarray, shape (N_source - 1, d)
        Markov-conditioned diagonal bridge volatility for each source interval.
    rng : np.random.Generator
        Random generator for bridge noise.

    Returns
    -------
    out : np.ndarray, shape (T, d)
        Path evaluated on target_times.
    """
    source_times = np.asarray(source_times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)

    if np.any(np.diff(source_times) <= 0):
        raise ValueError("source_times must be strictly increasing.")
    if np.any(np.diff(target_times) <= 0):
        raise ValueError("target_times must be strictly increasing.")
    if target_times[0] < source_times[0] or target_times[-1] > source_times[-1]:
        raise ValueError("target_times must lie within the source-time range.")

    n_source, d = grid_path.shape
    n_target = target_times.shape[0]
    out = np.empty((n_target, d), dtype=grid_path.dtype)

    target_pos = 0
    tol = 1e-12

    for i in range(n_source - 1):
        left = source_times[i]
        right = source_times[i + 1]

        current_t = left
        current_x = grid_path[i].astype(np.float64).copy()
        endpoint_y = grid_path[i + 1].astype(np.float64).copy()
        sigma_i = sigma_diag_by_interval[i].astype(np.float64)

        while target_pos < n_target and target_times[target_pos] <= right + tol:
            t = target_times[target_pos]

            if t < left - tol:
                raise RuntimeError("Internal target-time indexing error.")

            if abs(t - left) <= tol:
                out[target_pos] = grid_path[i]
                target_pos += 1
                continue

            if abs(t - right) <= tol:
                out[target_pos] = grid_path[i + 1]
                target_pos += 1
                break

            step_dt = t - current_t
            tau = right - current_t
            alpha = step_dt / tau
            var_factor = step_dt * (right - t) / tau

            z = rng.normal(size=d)
            current_x = current_x + alpha * (endpoint_y - current_x) + sigma_i * np.sqrt(var_factor) * z
            current_t = t

            out[target_pos] = current_x.astype(grid_path.dtype)
            target_pos += 1

    if target_pos != n_target:
        raise RuntimeError(
            f"Only filled {target_pos} of {n_target} target times. "
            "Check that source_times cover target_times."
        )

    return out


def bridge_batch_to_target_grid(syn_irregular, source_times, target_times, bridge_sigmas, seed=0):
    """
    Interpolate a batch of generated irregular-grid trajectories to target_times.

    Parameters
    ----------
    syn_irregular : np.ndarray, shape (M_simu, N_source, d)
        Generated endpoints on irregular grid.
    source_times : np.ndarray, shape (N_source,)
        Irregular grid.
    target_times : np.ndarray, shape (T,)
        Full regular grid.
    bridge_sigmas : np.ndarray, shape (M_simu, N_source - 1, d)
        Markov-conditioned bridge volatility returned by simulate_path_with_markov_sigma.
    seed : int
        Random seed for bridge interpolation.

    Returns
    -------
    syn_full : np.ndarray, shape (M_simu, T, d)
        Generated trajectories on the full target grid.
    """
    rng = np.random.default_rng(seed)

    M_simu, _, d = syn_irregular.shape
    T = target_times.shape[0]
    syn_full = np.empty((M_simu, T, d), dtype=syn_irregular.dtype)

    for s in range(M_simu):
        syn_full[s] = sample_bridge_to_target_grid(
            grid_path=syn_irregular[s],
            source_times=source_times,
            target_times=target_times,
            sigma_diag_by_interval=bridge_sigmas[s],
            rng=rng,
        )

    return syn_full


def run_shared_irregular_recovery_experiment(
    X_full,
    K=5,
    q=0.05,
    kernel_type="quartic",
    drop_frac=0.3,
    n_drop=None,
    M_simu=None,
    seed=0,
    bridge_scale=1.0,
):
    """
    Run the complete shared-irregular-grid recovery experiment.

    Parameters
    ----------
    X_full : np.ndarray, shape (M, T, d)
        Real data on the full regular grid.
    K : int
        Markov/history order.
    q : float
        q <= 1 is interpreted as a fraction of M; q > 1 as absolute neighbor count.
    kernel_type : str
        Kernel type: "quartic", "gaussian", "epanechnikov", or "triangular".
    drop_frac : float
        Fraction of interior time indices to drop if n_drop is None.
    n_drop : int or None
        Absolute number of interior time indices to drop.
    M_simu : int or None
        Number of synthetic samples. Defaults to M.
    seed : int
        Random seed.
    bridge_scale : float
        Multiplier for Markov-conditioned bridge volatility. Use values below 1
        for smoother interpolation.

    Returns
    -------
    result : dict
        Contains X_irregular, syn_irregular, syn_full, kept/dropped indices,
        times, labels, histories, bandwidths, bridge_sigmas, K_eff, and q_count.
    """
    if X_full.ndim != 3:
        raise ValueError(f"Expected X_full with shape (M, T, d), got {X_full.shape}.")

    # Ensures reproducibility for Numba code using np.random.
    np.random.seed(seed)

    M, T, d = X_full.shape

    if M_simu is None:
        M_simu = M

    keep_idx, drop_idx, full_times, irregular_times = make_shared_irregular_grid(
        T=T,
        drop_frac=drop_frac,
        n_drop=n_drop,
        seed=seed,
        keep_endpoints=True,
    )

    X_irregular = np.ascontiguousarray(X_full[:, keep_idx, :])
    N_irreg = X_irregular.shape[1]

    K_eff = min(K, N_irreg)
    q_count = resolve_q(q, M)

    syn_irregular, chosen_labels, hist_used, bandwidths, bridge_sigmas = simulate_path_with_markov_sigma(
        N=N_irreg,
        M=M,
        d=d,
        K=K_eff,
        X=X_irregular,
        q=q_count,
        M_simu=M_simu,
        times=irregular_times,
        kernel_type=kernel_type,
        bridge_scale=bridge_scale,
    )

    syn_full = bridge_batch_to_target_grid(
        syn_irregular=syn_irregular,
        source_times=irregular_times,
        target_times=full_times,
        bridge_sigmas=bridge_sigmas,
        seed=seed + 1,
    )

    return {
        "X_full": X_full,
        "X_irregular": X_irregular,
        "syn_irregular": syn_irregular,
        "syn_full": syn_full,
        "keep_idx": keep_idx,
        "drop_idx": drop_idx,
        "full_times": full_times,
        "irregular_times": irregular_times,
        "chosen_labels": chosen_labels,
        "hist_used": hist_used,
        "bandwidths": bandwidths,
        "bridge_sigmas": bridge_sigmas,
        "K_eff": K_eff,
        "q_count": q_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Shared-irregular-grid PATH recovery experiment")

    parser.add_argument("--data", type=str, default="stocks")
    parser.add_argument("--out_dir", type=str, default="synthetic_irregular_recovery")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--q", type=float, default=0.05)
    parser.add_argument("--kernel", type=str, default="quartic", choices=["quartic", "gaussian", "epanechnikov", "triangular"])
    parser.add_argument("--drop_frac", type=float, default=0.3)
    parser.add_argument("--n_drop", type=int, default=None)
    parser.add_argument("--M_simu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bridge_scale", type=float, default=1.0)

    args = parser.parse_args()

    
    X = np.load(f"data/{args.data}/train.npy")
    X_test = np.load(f'data/{args.data}/test.npy')
    M,_,_ = X_test.shape



    result = run_shared_irregular_recovery_experiment(
        X_full=X,
        K=args.K,
        q=args.q,
        kernel_type=args.kernel,
        drop_frac=args.drop_frac,
        n_drop=args.n_drop,
        M_simu=M,
        seed=args.seed,
        bridge_scale=args.bridge_scale,
    )

    # os.makedirs(args.out_dir, exist_ok=True)

    # syn_path = os.path.join(args.out_dir, "synthetic_recovered_full.npy")
    # diag_path = os.path.join(args.out_dir, "synthetic_recovered_diagnostics.npz")


   # np.save(syn_path, result["syn_full"])

    
    os.makedirs(f"synthetic/{args.data}",exist_ok=True)
    print(f'Samples save to synthetic/{args.data}/path_ada_irr_{args.drop_frac}.npy')
    np.save(f"synthetic/{args.data}/path_ada_irr_{args.drop_frac}.npy",result['syn_full'])
    # np.savez(
    #     diag_path,
    #     keep_idx=result["keep_idx"],
    #     drop_idx=result["drop_idx"],
    #     full_times=result["full_times"],
    #     irregular_times=result["irregular_times"],
    #     chosen_labels=result["chosen_labels"],
    #     hist_used=result["hist_used"],
    #     bandwidths=result["bandwidths"],
    #     bridge_sigmas=result["bridge_sigmas"],
    #     K_eff=result["K_eff"],
    #     q_count=result["q_count"],
    # )

    print("Real full shape:", result["X_full"].shape)
    print("Irregular train shape:", result["X_irregular"].shape)
    print("Synthetic irregular shape:", result["syn_irregular"].shape)
    print("Synthetic recovered full shape:", result["syn_full"].shape)
    print("Kept indices:", result["keep_idx"])
    print("Dropped indices:", result["drop_idx"])
    print("Resolved q_count:", result["q_count"])
    #print("Saved:", syn_)
    #print("Saved:", diag_path)


if __name__ == "__main__":
    main()
