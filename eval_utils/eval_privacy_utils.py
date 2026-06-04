"""
Privacy evaluation utilities for time series synthesis.

Two metrics:
  1. Nearest-Neighbour Distance (NND): compares min distances from synthetic
     trajectories to training set vs. held-out real test trajectories to
     training set. Systematic under-distance indicates memorisation.
  2. Subsequence Copying: longest near-matching contiguous block from training
     data found in each generated trajectory.
"""

from typing import Optional
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


# ── Pairwise distance helpers ─────────────────────────────────────────────────

def _flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1).astype(np.float32)


def _pairwise_l2(A: np.ndarray, B: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Pairwise L2: (M, P) × (N, P) → (M, N), batched over rows of A."""
    M, N = A.shape[0], B.shape[0]
    out = np.empty((M, N), dtype=np.float32)
    B_sq = np.sum(B ** 2, axis=1)
    for s in range(0, M, batch_size):
        e = min(s + batch_size, M)
        a = A[s:e]
        a_sq = np.sum(a ** 2, axis=1, keepdims=True)
        dist_sq = a_sq + B_sq[None, :] - 2.0 * (a @ B.T)
        np.clip(dist_sq, 0.0, None, out=dist_sq)
        out[s:e] = np.sqrt(dist_sq)
    return out


# ── Nearest-Neighbour Distance ────────────────────────────────────────────────

def nearest_neighbor_distances(
    X_query: np.ndarray,
    X_reference: np.ndarray,
    normalize: bool = True,
    batch_size: int = 256,
) -> np.ndarray:
    """
    For each query trajectory, return the L2 distance to its closest reference.

    Parameters
    ----------
    X_query, X_reference : (N, T, D)
    normalize            : divide by sqrt(T*D) → per-element scale

    Returns
    -------
    dists : (M,) float32
    """
    T, D = X_query.shape[1], X_query.shape[2]
    pw = _pairwise_l2(_flatten(X_query), _flatten(X_reference), batch_size=batch_size)
    dists = pw.min(axis=1)
    if normalize:
        dists = dists / float(np.sqrt(T * D))
    return dists


def privacy_nn_metrics(
    X_train: np.ndarray,
    X_test: np.ndarray,
    X_syn: np.ndarray,
    normalize: bool = True,
    batch_size: int = 256,
) -> dict:
    """
    Nearest-neighbour distance memorisation metrics.

    Compares:
      d_gen→train  = min distance from each synthetic trajectory to training set
      d_test→train = min distance from each held-out real trajectory to training set

    If d_gen→train << d_test→train the synthetic samples cluster unusually
    close to training data, indicating potential memorisation.

    Key scalars
    -----------
    nn_ratio_mean            : mean(d_gen) / mean(d_test).  Ideal ≈ 1; < 1 is suspicious.
    pct_gen_below_test_median: fraction of synthetic below the test median distance.
                               Ideal ≈ 0.5; >> 0.5 signals memorisation.
    ks_stat / ks_pvalue      : KS test between the two NN-distance distributions.
    """
    d_gen  = nearest_neighbor_distances(X_syn,  X_train, normalize=normalize, batch_size=batch_size)
    d_test = nearest_neighbor_distances(X_test, X_train, normalize=normalize, batch_size=batch_size)

    ks_stat, ks_pvalue = ks_2samp(d_gen, d_test)
    test_median = float(np.median(d_test))

    return {
        "nn_dist_gen_mean":          float(np.mean(d_gen)),
        "nn_dist_gen_median":        float(np.median(d_gen)),
        "nn_dist_gen_std":           float(np.std(d_gen)),
        "nn_dist_test_mean":         float(np.mean(d_test)),
        "nn_dist_test_median":       float(np.median(d_test)),
        "nn_dist_test_std":          float(np.std(d_test)),
        "nn_ratio_mean":             float(np.mean(d_gen) / (np.mean(d_test) + 1e-12)),
        "nn_ratio_median":           float(np.median(d_gen) / (test_median + 1e-12)),
        "ks_stat":                   float(ks_stat),
        "ks_pvalue":                 float(ks_pvalue),
        "pct_gen_below_test_median": float(np.mean(d_gen < test_median)),
    }


# ── Membership Inference Attack proxy ────────────────────────────────────────

def privacy_mia_metrics(
    X_train: np.ndarray,
    X_test: np.ndarray,
    X_syn: np.ndarray,
    normalize: bool = True,
    batch_size: int = 256,
) -> dict:
    """
    Membership Inference Attack (MIA) proxy via nearest-neighbour distance to
    the synthetic set.

    For each real trajectory x, compute d(x, X_syn) = min distance to any
    generated sample.  This is the attacker's feature: if the generator
    memorised training data, training trajectories (members) will have
    systematically smaller distances to the synthetic set than held-out real
    trajectories (non-members).

    Attack score = -d(x, X_syn)  (closer → more likely member).
    AUC is computed over (members ∪ non-members) with labels 1/0.

    Key scalars
    -----------
    mia_auc        : ROC-AUC of the attacker.  0.5 = random; 1.0 = perfect.
    mia_advantage  : 2*(AUC - 0.5), i.e. the attacker's edge above chance.
    """
    d_member    = nearest_neighbor_distances(X_train, X_syn, normalize=normalize, batch_size=batch_size)
    d_nonmember = nearest_neighbor_distances(X_test,  X_syn, normalize=normalize, batch_size=batch_size)

    scores = np.concatenate([-d_member, -d_nonmember])
    labels = np.concatenate([np.ones(len(X_train)), np.zeros(len(X_test))])

    auc = float(roc_auc_score(labels, scores))

    return {
        "mia_auc":            auc,
        "mia_advantage":      float(2.0 * (auc - 0.5)),
        "d_member_mean":      float(np.mean(d_member)),
        "d_member_median":    float(np.median(d_member)),
        "d_nonmember_mean":   float(np.mean(d_nonmember)),
        "d_nonmember_median": float(np.median(d_nonmember)),
    }


# ── Subsequence copying ───────────────────────────────────────────────────────

def _lmax_batch(
    X_syn_batch: np.ndarray,   # (B, T, D)
    X_train: np.ndarray,        # (N, T, D)
    tau: float,
    min_len: int,
) -> np.ndarray:
    """
    Vectorised L_max for a batch of synthetic trajectories vs. all training
    trajectories with all 2T-1 temporal offsets.  Returns (B,) int32.
    """
    B, T_g, _ = X_syn_batch.shape
    N, T_t    = X_train.shape[0], X_train.shape[1]

    # diff[b, n, t_g, t_t] = ||syn[b, t_g] - train[n, t_t]||₂
    diff = np.linalg.norm(
        X_syn_batch[:, None, :, None, :]   # (B, 1, T_g, 1, D)
        - X_train[None, :, None, :, :],    # (1, N, 1, T_t, D)
        axis=-1,
    ).astype(np.float32)                   # (B, N, T_g, T_t)

    L_max = np.zeros(B, dtype=np.int32)

    # Process longest diagonals first so early-exit triggers sooner.
    offsets = sorted(range(-(T_g - 1), T_t), key=abs)  # 0, ±1, ±2, ...

    for d in offsets:
        # diagonal(offset=d, axis1=2, axis2=3) → (B, N, L)
        diag = np.diagonal(diff, offset=d, axis1=2, axis2=3).copy()   # copy for contiguity
        L = diag.shape[-1]

        # skip if this diagonal can't improve any sample's best
        if L < min_len or L <= int(L_max.max()):
            continue

        # prefix sums along the subsequence axis
        cs = np.empty((B, N, L + 1), dtype=np.float32)
        cs[:, :, 0] = 0.0
        np.cumsum(diag, axis=-1, out=cs[:, :, 1:])

        for l in range(L, min_len - 1, -1):
            if np.all(L_max >= l):
                break   # every sample in batch already beats this length

            # window sums for all (n, start) pairs: (B, N, W)
            window_sums = cs[:, :, l:] - cs[:, :, : L - l + 1]
            # any match across (n, start) for each batch element
            found = np.any(window_sums <= tau * l, axis=(1, 2))   # (B,)
            improve = found & (L_max < l)
            L_max[improve] = l

    return L_max


def _lmax_batch_aligned(
    X_syn_batch: np.ndarray,   # (B, T, D)
    X_train: np.ndarray,        # (N, T, D)
    tau: float,
    min_len: int,
) -> np.ndarray:
    """
    Fast variant: only checks the d=0 (same-time-alignment) diagonal.
    Per-timestep L2: diff[b, n, t] = ||syn[b, t] - train[n, t]||₂.  (B, N, T)
    """
    B, T, _ = X_syn_batch.shape
    N = X_train.shape[0]

    # (B, N, T) — d=0 only
    diag = np.linalg.norm(
        X_syn_batch[:, None, :, :]   # (B, 1, T, D)
        - X_train[None, :, :, :],    # (1, N, T, D)
        axis=-1,
    ).astype(np.float32)

    L_max = np.zeros(B, dtype=np.int32)
    cs = np.empty((B, N, T + 1), dtype=np.float32)
    cs[:, :, 0] = 0.0
    np.cumsum(diag, axis=-1, out=cs[:, :, 1:])

    for l in range(T, min_len - 1, -1):
        if np.all(L_max >= l):
            break
        window_sums = cs[:, :, l:] - cs[:, :, : T - l + 1]
        found = np.any(window_sums <= tau * l, axis=(1, 2))
        improve = found & (L_max < l)
        L_max[improve] = l

    return L_max


def auto_tau(
    X_train: np.ndarray,
    quantile: float = 0.1,
    n_pairs: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Data-adaptive tau: a quantile of mean per-timestep L2 distances between
    random training trajectory pairs.

    At quantile=0.1 the threshold captures subsequences that are closer than
    90 % of training-pair distances — a conservative memorisation signal.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    N = X_train.shape[0]
    n_pairs = min(n_pairs, max(1, N * (N - 1) // 2))

    i_idx = rng.integers(0, N, size=n_pairs)
    j_idx = rng.integers(0, N, size=n_pairs)
    same = i_idx == j_idx
    j_idx[same] = (j_idx[same] + 1) % N

    # vectorised: (n_pairs, T, D) → (n_pairs, T) norms → (n_pairs,) means
    dists = np.linalg.norm(
        X_train[i_idx] - X_train[j_idx], axis=-1
    ).mean(axis=-1)
    return float(np.quantile(dists, quantile))


def subsequence_copy_lengths(
    X_syn: np.ndarray,
    X_train: np.ndarray,
    tau: float,
    min_len: int = 2,
    n_train_max: Optional[int] = None,
    n_syn_max: Optional[int] = None,
    batch_size: int = 256,
    aligned_only: bool = False,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    For each synthetic trajectory compute L_max: the longest contiguous
    subsequence that near-matches any training trajectory at threshold tau.

    Distance per subsequence = mean per-timestep L2 norm.

    Parameters
    ----------
    tau          : per-timestep L2 threshold
    n_train_max  : subsample training set (speed; default uses all)
    n_syn_max    : subsample synthetic set (speed; default uses all)
    batch_size   : synthetic samples per vectorised batch.
                   Memory ≈ batch_size × n_train × T² × 4 bytes.
                   Reduce for large T (e.g. 32 for T=100, N=200).
    aligned_only : only check d=0 offset (much faster; less general)

    Returns
    -------
    L_max : (M,) int32  (or (n_syn_max,) if subsampled)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if n_train_max is not None and n_train_max < X_train.shape[0]:
        X_train = X_train[rng.choice(X_train.shape[0], size=n_train_max, replace=False)]

    if n_syn_max is not None and n_syn_max < X_syn.shape[0]:
        X_syn = X_syn[rng.choice(X_syn.shape[0], size=n_syn_max, replace=False)]

    M = X_syn.shape[0]
    L_max = np.zeros(M, dtype=np.int32)
    fn = _lmax_batch_aligned if aligned_only else _lmax_batch
    n_batches = (M + batch_size - 1) // batch_size

    for b in range(n_batches):
        s, e = b * batch_size, min((b + 1) * batch_size, M)
        L_max[s:e] = fn(X_syn[s:e], X_train, tau=tau, min_len=min_len)
        if verbose:
            print(f"    subsequence: {e}/{M}")

    return L_max


def privacy_subseq_metrics(
    X_syn: np.ndarray,
    X_train: np.ndarray,
    tau: float,
    min_len: int = 2,
    n_train_max: Optional[int] = None,
    n_syn_max: Optional[int] = None,
    batch_size: int = 256,
    aligned_only: bool = False,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> dict:
    """
    Summarise subsequence-copying lengths.

    subseq_lmax_norm_mean = mean(L_max) / T.  Ideal ≈ 0; close to 1 indicates
    that generated trajectories closely track entire training sequences.
    subseq_lmax_frac = fraction with any copy of length ≥ min_len detected.
    """
    T = X_syn.shape[1]
    L_max = subsequence_copy_lengths(
        X_syn, X_train, tau=tau, min_len=min_len,
        n_train_max=n_train_max, n_syn_max=n_syn_max,
        batch_size=batch_size, aligned_only=aligned_only,
        rng=rng, verbose=verbose,
    )
    return {
        "subseq_lmax_mean":      float(np.mean(L_max)),
        "subseq_lmax_max":       int(np.max(L_max)),
        "subseq_lmax_p75":       float(np.percentile(L_max, 75)),
        "subseq_lmax_p90":       float(np.percentile(L_max, 90)),
        "subseq_lmax_norm_mean": float(np.mean(L_max) / T),
        "subseq_lmax_frac":      float(np.mean(L_max >= min_len)),
        "tau":                   float(tau),
    }
