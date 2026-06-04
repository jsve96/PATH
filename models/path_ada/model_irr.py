import numpy as np
import numba as nb
from tqdm import tqdm


KERNEL_QUARTIC = 0
KERNEL_GAUSSIAN = 1
KERNEL_EPANECHNIKOV = 2
KERNEL_TRIANGULAR = 3

KERNEL_NAME_TO_CODE = {
    "quartic": KERNEL_QUARTIC,
    "gaussian": KERNEL_GAUSSIAN,
    "epanechnikov": KERNEL_EPANECHNIKOV,
    "triangular": KERNEL_TRIANGULAR,
}


def parse_kernel_type(kernel_type):
    if isinstance(kernel_type, str):
        key = kernel_type.lower()
        if key not in KERNEL_NAME_TO_CODE:
            valid = ", ".join(KERNEL_NAME_TO_CODE.keys())
            raise ValueError(f"Unknown kernel_type={kernel_type!r}. Valid values: {valid}.")
        return KERNEL_NAME_TO_CODE[key]

    kernel_code = int(kernel_type)
    if kernel_code not in (
        KERNEL_QUARTIC,
        KERNEL_GAUSSIAN,
        KERNEL_EPANECHNIKOV,
        KERNEL_TRIANGULAR,
    ):
        raise ValueError(f"Unknown integer kernel_type={kernel_code}.")
    return kernel_code


@nb.njit(cache=True)
def kernel_value_from_ratio_sq(ratio_sq, kernel_type):
    """
    Kernel value K(r), where ratio_sq = r^2 and r = rho / h.
    This matches weights of the form K(rho_i(z, Z_i^(m)) / h_i,q(z)).
    """
    if kernel_type == KERNEL_QUARTIC:
        if ratio_sq >= 1.0:
            return 0.0
        val = 1.0 - ratio_sq
        return val * val

    elif kernel_type == KERNEL_GAUSSIAN:
        return np.exp(-0.5 * ratio_sq)

    elif kernel_type == KERNEL_EPANECHNIKOV:
        if ratio_sq >= 1.0:
            return 0.0
        return 1.0 - ratio_sq

    elif kernel_type == KERNEL_TRIANGULAR:
        if ratio_sq >= 1.0:
            return 0.0
        return 1.0 - np.sqrt(ratio_sq)

    # Default to quartic.
    if ratio_sq >= 1.0:
        return 0.0
    val = 1.0 - ratio_sq
    return val * val


@nb.njit(cache=True)
def sample_categorical(weights):
    u = np.random.random()
    cdf = 0.0
    last = weights.shape[0] - 1

    for m in range(weights.shape[0]):
        cdf += weights[m]
        if u <= cdf:
            return m

    return last


@nb.njit(cache=True)
def qth_smallest_value(values, q):
    """
    Return the q-th smallest entry of values, using one-based q.
    This avoids relying on np.partition support and keeps the code Numba-stable.
    Complexity is O(q M), fine for moderate M. For very large M, replace by np.partition.
    """
    M = values.shape[0]
    used = np.zeros(M, dtype=np.bool_)
    qth = 0.0

    for _ in range(q):
        best_idx = -1
        best_val = np.inf

        for m in range(M):
            if not used[m] and values[m] < best_val:
                best_val = values[m]
                best_idx = m

        used[best_idx] = True
        qth = best_val

    return qth


@nb.njit(cache=True)
def stacked_history_sqdistances(i, hist_len, X, path, distances):
    """
    Compute squared stacked Euclidean history distances

        rho_i(z, Z_i^(m))^2
        = sum_{j=i-hist_len+1}^{i} ||path[j] - X[m,j]||^2.

    Here i is zero-based and corresponds to the transition from i to i+1.
    """
    M = X.shape[0]
    d = X.shape[2]
    start = i - hist_len + 1

    for m in range(M):
        acc = 0.0
        for j in range(start, i + 1):
            for r in range(d):
                diff = path[j, r] - X[m, j, r]
                acc += diff * diff
        distances[m] = acc


@nb.njit(cache=True)
def adaptive_knn_history_weights(i, K, X, path, q, kernel_type):
    """
    Compute normalized weights for transition [t_i, t_{i+1}] using
    stacked-history distance and the q-th nearest-neighbor bandwidth.

    The implemented estimator is

        w_i^(m)(z) = K( rho_i(z, Z_i^(m)) / h_{i,q}(z) ),

    where h_{i,q}(z) is the q-th nearest-neighbor radius among the historical
    stacked histories at step i.

    Parameters
    ----------
    i : int
        Zero-based transition index.
    K : int
        Markov/history order.
    X : np.ndarray, shape (M, N, d)
        Historical trajectories.
    path : np.ndarray, shape (N, d)
        Partially generated path.
    q : int
        Neighbor parameter. Must satisfy 1 <= q <= M before entering this function.
    kernel_type : int
        Kernel code.

    Returns
    -------
    weights : np.ndarray, shape (M,)
        Normalized transition weights.
    hist_len_used : int
        Effective history length min(K, i + 1).
    bandwidth : float
        q-th nearest-neighbor radius h_{i,q}(z).
    """
    M = X.shape[0]

    hist_len = K
    if i + 1 < hist_len:
        hist_len = i + 1

    distances_sq = np.empty(M, dtype=np.float64)
    stacked_history_sqdistances(i, hist_len, X, path, distances_sq)

    h_sq = qth_smallest_value(distances_sq, q)
    weights = np.zeros(M, dtype=np.float64)

    # Degenerate case: the q-th neighbor has zero distance. This occurs when the
    # query history exactly matches at least q historical histories. We place a
    # uniform distribution on the exact matches to avoid division by zero.
    if h_sq <= 0.0:
        count = 0
        for m in range(M):
            if distances_sq[m] <= 0.0:
                count += 1

        if count > 0:
            inv_count = 1.0 / count
            for m in range(M):
                if distances_sq[m] <= 0.0:
                    weights[m] = inv_count
            return weights, hist_len, 0.0

        # Extremely defensive fallback. Should not be reached.
        inv_M = 1.0 / M
        for m in range(M):
            weights[m] = inv_M
        return weights, hist_len, 0.0

    total = 0.0
    for m in range(M):
        ratio_sq = distances_sq[m] / h_sq
        w = kernel_value_from_ratio_sq(ratio_sq, kernel_type)
        weights[m] = w
        total += w

    if total > 0.0:
        inv_total = 1.0 / total
        for m in range(M):
            weights[m] *= inv_total
        return weights, hist_len, np.sqrt(h_sq)

    # Compactly supported kernels can assign zero weight to boundary points,
    # especially for small q. If all kernel weights vanish, fall back to a
    # uniform distribution over the q-nearest-neighbor ball.
    count = 0
    for m in range(M):
        if distances_sq[m] <= h_sq:
            count += 1

    if count > 0:
        inv_count = 1.0 / count
        for m in range(M):
            if distances_sq[m] <= h_sq:
                weights[m] = inv_count
        return weights, hist_len, np.sqrt(h_sq)

    # Defensive fallback. Should not be reached.
    inv_M = 1.0 / M
    for m in range(M):
        weights[m] = inv_M
    return weights, hist_len, np.sqrt(h_sq)


@nb.njit(cache=True)
def simulate_exact_latent_endpoint_grid_core(N, K, X, q, kernel_type, x0):
    """
    Simulate one path on the observation grid using the adaptive empirical
    conditional transition kernel.

    Returns
    -------
    path : np.ndarray, shape (N, d)
    chosen_labels : np.ndarray, shape (N - 1,)
        Index m of the historical endpoint selected at each transition.
    hist_used : np.ndarray, shape (N - 1,)
        History length min(K, i + 1) used at each transition.
    bandwidths : np.ndarray, shape (N - 1,)
        q-th nearest-neighbor radius h_{i,q}(z) used at each transition.
    """
    d = X.shape[2]
    n_transitions = N - 1

    path = np.empty((N, d), dtype=X.dtype)
    chosen_labels = np.empty(n_transitions, dtype=np.int64)
    hist_used = np.empty(n_transitions, dtype=np.int64)
    bandwidths = np.empty(n_transitions, dtype=np.float64)

    for r in range(d):
        path[0, r] = x0[r]

    for i in range(n_transitions):
        weights, hist_len, bandwidth = adaptive_knn_history_weights(
            i, K, X, path, q, kernel_type
        )
        m = sample_categorical(weights)

        chosen_labels[i] = m
        hist_used[i] = hist_len
        bandwidths[i] = bandwidth

        for r in range(d):
            path[i + 1, r] = X[m, i + 1, r]

    return path, chosen_labels, hist_used, bandwidths


def simulate_exact_latent_endpoint_grid(
    N,
    M,
    d,
    K,
    X,
    q,
    kernel_type="quartic",
    x0=None,
    sample_initial=True,
):
    """
    Simulate one grid-level path.

    Parameters
    ----------
    N : int
        Number of time points in each path.
    M : int
        Number of historical trajectories.
    d : int
        State dimension.
    K : int
        Markov/history order.
    X : np.ndarray, shape (M, N, d)
        Historical trajectories.
    q : int
        Neighbor parameter used to compute h_{i,q}(z).
    kernel_type : {"quartic", "gaussian", "epanechnikov", "triangular"}
        Kernel used on the normalized stacked distance.
    x0 : np.ndarray, shape (d,), optional
        Initial state. If omitted, sampled from the empirical initial distribution.
    sample_initial : bool
        If True and x0 is omitted, sample the initial state uniformly from X[:, 0, :].
        If False and x0 is omitted, use X[0, 0, :].

    Returns
    -------
    path : np.ndarray, shape (N, d)
    chosen_labels : np.ndarray, shape (N - 1,)
    hist_used : np.ndarray, shape (N - 1,)
    bandwidths : np.ndarray, shape (N - 1,)
    """
    if X.shape != (M, N, d):
        raise ValueError(f"Expected X.shape == ({M}, {N}, {d}), got {X.shape}.")
    if not (1 <= q <= M):
        raise ValueError(f"q must satisfy 1 <= q <= M={M}, got q={q}.")
    if K < 1:
        raise ValueError(f"K must be at least 1, got K={K}.")

    kernel_code = parse_kernel_type(kernel_type)

    if x0 is None:
        if sample_initial:
            m0 = np.random.randint(M)
            x0_ = X[m0, 0].copy()
        else:
            x0_ = X[0, 0].copy()
    else:
        x0_ = np.asarray(x0, dtype=X.dtype).copy()
        if x0_.shape != (d,):
            raise ValueError(f"x0 must have shape ({d},), got {x0_.shape}.")

    return simulate_exact_latent_endpoint_grid_core(
        N, K, X, q, kernel_code, x0_
    )


@nb.njit(cache=True)
def interpolate_exact_bridges(grid_path, deltati, N_pi, sigma=1.0):
    """
    Brownian-bridge interpolation between successive grid endpoints.

    This implementation uses scalar sigma, corresponding to covariance
    sigma^2 I_d for the Brownian reference process.
    """
    N = grid_path.shape[0] - 1
    d = grid_path.shape[1]

    out = np.empty((N * N_pi + 1, d), dtype=grid_path.dtype)

    for r in range(d):
        out[0, r] = grid_path[0, r]

    idx = 0
    dt = deltati / N_pi

    for i in range(N):
        x = np.empty(d, dtype=grid_path.dtype)
        y = np.empty(d, dtype=grid_path.dtype)

        for r in range(d):
            x[r] = grid_path[i, r]
            y[r] = grid_path[i + 1, r]

        for k in range(1, N_pi + 1):
            idx += 1

            if k == N_pi:
                for r in range(d):
                    out[idx, r] = y[r]
                continue

            elapsed = (k - 1) * dt
            tau = deltati - elapsed

            alpha = dt / tau
            var_factor = dt * (tau - dt) / tau
            std = sigma * np.sqrt(var_factor)

            for r in range(d):
                z = np.random.normal()
                x[r] = x[r] + alpha * (y[r] - x[r]) + std * z
                out[idx, r] = x[r]

    return out


def simulate_path(
    N,
    M,
    d,
    K,
    X,
    q,
    M_simu,
    kernel_type="quartic",
    x0=None,
    sample_initial=True,
):
    """
    Simulate multiple grid-level synthetic paths.

    Parameters
    ----------
    N : int
        Number of time points in each path.
    M : int
        Number of historical trajectories.
    d : int
        State dimension.
    K : int
        Markov/history order.
    X : np.ndarray, shape (M, N, d)
        Historical trajectories.
    q : int
        Neighbor parameter used to compute the local bandwidth h_{i,q}(z).
    M_simu : int
        Number of synthetic trajectories to generate.

    Returns
    -------
    data_sb : np.ndarray, shape (M_simu, N, d)
    chosen_labels : np.ndarray, shape (M_simu, N - 1)
    hist_used : np.ndarray, shape (M_simu, N - 1)
    bandwidths : np.ndarray, shape (M_simu, N - 1)
    """
    if X.shape != (M, N, d):
        raise ValueError(f"Expected X.shape == ({M}, {N}, {d}), got {X.shape}.")
    if not (1 <= q <= M):
        raise ValueError(f"q must satisfy 1 <= q <= M={M}, got q={q}.")
    if K < 1:
        raise ValueError(f"K must be at least 1, got K={K}.")

    kernel_code = parse_kernel_type(kernel_type)

    data_sb = np.empty((M_simu, N, d), dtype=X.dtype)
    chosen_labels = np.empty((M_simu, N - 1), dtype=np.int64)
    hist_used = np.empty((M_simu, N - 1), dtype=np.int64)
    bandwidths = np.empty((M_simu, N - 1), dtype=np.float64)

    for k in tqdm(range(M_simu)):
        if x0 is None:
            if sample_initial:
                m0 = np.random.randint(M)
                x0_k = X[m0, 0].copy()
            else:
                x0_k = X[0, 0].copy()
        else:
            x0_k = np.asarray(x0, dtype=X.dtype).copy()
            if x0_k.shape != (d,):
                raise ValueError(f"x0 must have shape ({d},), got {x0_k.shape}.")

        path, labels_k, hist_k, bandwidths_k = simulate_exact_latent_endpoint_grid_core(
            N, K, X, q, kernel_code, x0_k
        )

        data_sb[k, :, :] = path
        chosen_labels[k, :] = labels_k
        hist_used[k, :] = hist_k
        bandwidths[k, :] = bandwidths_k

    return data_sb, chosen_labels, hist_used, bandwidths


# Optional backward-compatible alias with explicit parameter names.
def simulate_path_knn(
    N,
    M,
    d,
    markov_order,
    X,
    q,
    M_simu,
    kernel_type="quartic",
    x0=None,
    sample_initial=True,
):
    return simulate_path(
        N=N,
        M=M,
        d=d,
        K=markov_order,
        X=X,
        q=q,
        M_simu=M_simu,
        kernel_type=kernel_type,
        x0=x0,
        sample_initial=sample_initial,
    )


def format_bytes(n):
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    n = float(n)
    for unit in units:
        if n < 1024:
            return f"{n:,.2f} {unit}"
        n /= 1024
    return f"{n:,.2f} PiB"


def estimate_simulate_path_memory(N, M, d, K, X, M_simu, include_X=True):
    x_itemsize = X.dtype.itemsize
    int64_size = np.dtype(np.int64).itemsize
    float64_size = np.dtype(np.float64).itemsize

    data_sb_bytes = M_simu * N * d * x_itemsize
    chosen_labels_bytes = M_simu * (N - 1) * int64_size
    hist_used_bytes = M_simu * (N - 1) * int64_size
    bandwidths_bytes = M_simu * (N - 1) * float64_size

    returned_total = (
        data_sb_bytes
        + chosen_labels_bytes
        + hist_used_bytes
        + bandwidths_bytes
    )

    path_bytes = N * d * x_itemsize
    labels_k_bytes = (N - 1) * int64_size
    hist_k_bytes = (N - 1) * int64_size
    bandwidths_k_bytes = (N - 1) * float64_size
    distances_sq_bytes = M * float64_size
    weights_bytes = M * float64_size
    used_bytes = M * np.dtype(np.bool_).itemsize

    temporary_peak = (
        path_bytes
        + labels_k_bytes
        + hist_k_bytes
        + bandwidths_k_bytes
        + distances_sq_bytes
        + weights_bytes
        + used_bytes
    )

    x_bytes = X.nbytes

    print("Estimated memory used by simulate_path")
    print("--------------------------------------")
    print(f"X input array:             {format_bytes(x_bytes)}")
    print()
    print("Returned arrays:")
    print(f"  data_sb:                 {format_bytes(data_sb_bytes)}")
    print(f"  chosen_labels:           {format_bytes(chosen_labels_bytes)}")
    print(f"  hist_used:               {format_bytes(hist_used_bytes)}")
    print(f"  bandwidths:              {format_bytes(bandwidths_bytes)}")
    print(f"  returned total:          {format_bytes(returned_total)}")
    print()
    print("Temporary peak, approx:")
    print(f"  one path:                {format_bytes(path_bytes)}")
    print(f"  labels_k:                {format_bytes(labels_k_bytes)}")
    print(f"  hist_k:                  {format_bytes(hist_k_bytes)}")
    print(f"  bandwidths_k:            {format_bytes(bandwidths_k_bytes)}")
    print(f"  distances_sq:            {format_bytes(distances_sq_bytes)}")
    print(f"  weights:                 {format_bytes(weights_bytes)}")
    print(f"  qth-selection mask:      {format_bytes(used_bytes)}")
    print(f"  temporary peak:          {format_bytes(temporary_peak)}")
    print()
    print(f"Estimated extra memory:    {format_bytes(returned_total + temporary_peak)}")

    if include_X:
        print(f"Estimated total incl. X:   {format_bytes(x_bytes + returned_total + temporary_peak)}")

    return {
        "X": x_bytes,
        "data_sb": data_sb_bytes,
        "chosen_labels": chosen_labels_bytes,
        "hist_used": hist_used_bytes,
        "bandwidths": bandwidths_bytes,
        "returned_total": returned_total,
        "temporary_peak": temporary_peak,
        "extra_memory": returned_total + temporary_peak,
        "total_including_X": x_bytes + returned_total + temporary_peak,
    }


# -----------------------------------------------------------------------------
# Markov-conditioned bridge volatility extension
# -----------------------------------------------------------------------------

@nb.njit(cache=True)
def markov_conditional_increment_sigma_diag(i, X, weights, delta_t, eps=1e-8, bridge_scale=1.0):
    """
    Estimate a Markov-conditioned diagonal bridge volatility for interval i.

    Uses the same normalized kernel weights used for endpoint generation:

        pi_i^(m)(z_i), m=1,...,M.

    Historical increments are

        dX_i^(m) = X[m, i+1, :] - X[m, i, :].

    The diagonal diffusion estimate is

        sigma_{i,r}^2(z_i)
        = (1 / delta_t) * sum_m pi_m * (dX_{i,r}^(m) - mean_r)^2.

    Parameters
    ----------
    i : int
        Zero-based interval index.
    X : np.ndarray, shape (M, N, d)
        Historical trajectories.
    weights : np.ndarray, shape (M,)
        Normalized Markov/kernel weights for the current generated history.
    delta_t : float
        Interval length t_{i+1} - t_i.
    eps : float
        Variance floor for numerical stability.
    bridge_scale : float
        Optional multiplier for bridge volatility. Use < 1 for smoother bridges.

    Returns
    -------
    sigma_diag : np.ndarray, shape (d,)
        Per-dimension local bridge volatility for interval i.
    """
    M = X.shape[0]
    d = X.shape[2]

    mean = np.zeros(d, dtype=np.float64)
    sigma_diag = np.empty(d, dtype=np.float64)

    # Weighted mean of historical increments.
    for m in range(M):
        wm = weights[m]
        for r in range(d):
            inc = X[m, i + 1, r] - X[m, i, r]
            mean[r] += wm * inc

    # Weighted diagonal covariance of historical increments.
    for r in range(d):
        var_r = 0.0
        for m in range(M):
            inc = X[m, i + 1, r] - X[m, i, r]
            diff = inc - mean[r]
            var_r += weights[m] * diff * diff

        # Convert increment variance to diffusion variance by dividing by delta_t.
        val = var_r / delta_t
        if val < eps:
            val = eps
        sigma_diag[r] = bridge_scale * np.sqrt(val)

    return sigma_diag


@nb.njit(cache=True)
def simulate_exact_latent_endpoint_grid_core_with_markov_sigma(
    N,
    K,
    X,
    q,
    kernel_type,
    x0,
    times,
    bridge_eps=1e-8,
    bridge_scale=1.0,
):
    """
    Simulate one endpoint path and estimate Markov-conditioned bridge sigmas.

    Returns
    -------
    path : np.ndarray, shape (N, d)
    chosen_labels : np.ndarray, shape (N - 1,)
    hist_used : np.ndarray, shape (N - 1,)
    bandwidths : np.ndarray, shape (N - 1,)
    bridge_sigmas : np.ndarray, shape (N - 1, d)
        Local diagonal bridge volatility sigma_i(z_i) for each interval.
    """
    d = X.shape[2]
    n_transitions = N - 1

    path = np.empty((N, d), dtype=X.dtype)
    chosen_labels = np.empty(n_transitions, dtype=np.int64)
    hist_used = np.empty(n_transitions, dtype=np.int64)
    bandwidths = np.empty(n_transitions, dtype=np.float64)
    bridge_sigmas = np.empty((n_transitions, d), dtype=np.float64)

    for r in range(d):
        path[0, r] = x0[r]

    for i in range(n_transitions):
        weights, hist_len, bandwidth = adaptive_knn_history_weights(
            i, K, X, path, q, kernel_type
        )

        delta_t = times[i + 1] - times[i]
        sigma_i = markov_conditional_increment_sigma_diag(
            i, X, weights, delta_t, bridge_eps, bridge_scale
        )

        m = sample_categorical(weights)

        chosen_labels[i] = m
        hist_used[i] = hist_len
        bandwidths[i] = bandwidth

        for r in range(d):
            bridge_sigmas[i, r] = sigma_i[r]
            path[i + 1, r] = X[m, i + 1, r]

    return path, chosen_labels, hist_used, bandwidths, bridge_sigmas


def simulate_path_with_markov_sigma(
    N,
    M,
    d,
    K,
    X,
    q,
    M_simu,
    times=None,
    kernel_type="quartic",
    x0=None,
    sample_initial=True,
    bridge_eps=1e-8,
    bridge_scale=1.0,
):
    """
    Simulate multiple endpoint paths and local Markov-conditioned bridge sigmas.

    Use this instead of simulate_path when you want Brownian-bridge interpolation
    with history-dependent volatility.

    Returns
    -------
    data_sb : np.ndarray, shape (M_simu, N, d)
    chosen_labels : np.ndarray, shape (M_simu, N - 1)
    hist_used : np.ndarray, shape (M_simu, N - 1)
    bandwidths : np.ndarray, shape (M_simu, N - 1)
    bridge_sigmas : np.ndarray, shape (M_simu, N - 1, d)
    """
    if X.shape != (M, N, d):
        raise ValueError(f"Expected X.shape == ({M}, {N}, {d}), got {X.shape}.")
    if not (1 <= q <= M):
        raise ValueError(f"q must satisfy 1 <= q <= M={M}, got q={q}.")
    if K < 1:
        raise ValueError(f"K must be at least 1, got K={K}.")

    if times is None:
        times_ = np.arange(N, dtype=np.float64)
    else:
        times_ = np.asarray(times, dtype=np.float64)
        if times_.shape != (N,):
            raise ValueError(f"times must have shape ({N},), got {times_.shape}.")
        if np.any(np.diff(times_) <= 0):
            raise ValueError("times must be strictly increasing.")

    if bridge_scale < 0:
        raise ValueError("bridge_scale must be nonnegative.")
    if bridge_eps < 0:
        raise ValueError("bridge_eps must be nonnegative.")

    kernel_code = parse_kernel_type(kernel_type)

    data_sb = np.empty((M_simu, N, d), dtype=X.dtype)
    chosen_labels = np.empty((M_simu, N - 1), dtype=np.int64)
    hist_used = np.empty((M_simu, N - 1), dtype=np.int64)
    bandwidths = np.empty((M_simu, N - 1), dtype=np.float64)
    bridge_sigmas = np.empty((M_simu, N - 1, d), dtype=np.float64)

    for s in tqdm(range(M_simu)):
        if x0 is None:
            if sample_initial:
                m0 = np.random.randint(M)
                x0_s = X[m0, 0].copy()
            else:
                x0_s = X[0, 0].copy()
        else:
            x0_s = np.asarray(x0, dtype=X.dtype).copy()
            if x0_s.shape != (d,):
                raise ValueError(f"x0 must have shape ({d},), got {x0_s.shape}.")

        path, labels_s, hist_s, bandwidths_s, sigmas_s = (
            simulate_exact_latent_endpoint_grid_core_with_markov_sigma(
                N=N,
                K=K,
                X=X,
                q=q,
                kernel_type=kernel_code,
                x0=x0_s,
                times=times_,
                bridge_eps=bridge_eps,
                bridge_scale=bridge_scale,
            )
        )

        data_sb[s, :, :] = path
        chosen_labels[s, :] = labels_s
        hist_used[s, :] = hist_s
        bandwidths[s, :] = bandwidths_s
        bridge_sigmas[s, :, :] = sigmas_s

    return data_sb, chosen_labels, hist_used, bandwidths, bridge_sigmas


@nb.njit(cache=True)
def interpolate_exact_bridges_irregular_markov_sigma(grid_path, times, N_pi, sigma_diag_by_interval):
    """
    Brownian-bridge interpolation with interval- and state-dependent diagonal sigma.

    Parameters
    ----------
    grid_path : np.ndarray, shape (N, d)
        Generated endpoint path.
    times : np.ndarray, shape (N,)
        Strictly increasing observation times. Can be irregular.
    N_pi : int
        Number of substeps per interval.
    sigma_diag_by_interval : np.ndarray, shape (N - 1, d)
        Local bridge volatility returned by simulate_path_with_markov_sigma
        for one generated path.

    Returns
    -------
    out : np.ndarray, shape ((N - 1) * N_pi + 1, d)
    out_times : np.ndarray, shape ((N - 1) * N_pi + 1,)
    """
    N_total = grid_path.shape[0]
    n_intervals = N_total - 1
    d = grid_path.shape[1]

    out = np.empty((n_intervals * N_pi + 1, d), dtype=grid_path.dtype)
    out_times = np.empty(n_intervals * N_pi + 1, dtype=times.dtype)

    for r in range(d):
        out[0, r] = grid_path[0, r]
    out_times[0] = times[0]

    idx = 0

    for i in range(n_intervals):
        delta_t = times[i + 1] - times[i]
        dt = delta_t / N_pi

        x = np.empty(d, dtype=grid_path.dtype)
        y = np.empty(d, dtype=grid_path.dtype)

        for r in range(d):
            x[r] = grid_path[i, r]
            y[r] = grid_path[i + 1, r]

        for k in range(1, N_pi + 1):
            idx += 1
            out_times[idx] = times[i] + k * dt

            if k == N_pi:
                for r in range(d):
                    out[idx, r] = y[r]
                continue

            elapsed = (k - 1) * dt
            tau = delta_t - elapsed

            alpha = dt / tau
            var_factor = dt * (tau - dt) / tau
            sqrt_var_factor = np.sqrt(var_factor)

            for r in range(d):
                z = np.random.normal()
                std = sigma_diag_by_interval[i, r] * sqrt_var_factor
                x[r] = x[r] + alpha * (y[r] - x[r]) + std * z
                out[idx, r] = x[r]

    return out, out_times


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


__all__ = [
    "KERNEL_QUARTIC",
    "KERNEL_GAUSSIAN",
    "KERNEL_EPANECHNIKOV",
    "KERNEL_TRIANGULAR",
    "parse_kernel_type",
    "resolve_q",
    "adaptive_knn_history_weights",
    "simulate_exact_latent_endpoint_grid",
    "simulate_path",
    "simulate_path_knn",
    "simulate_path_with_markov_sigma",
    "markov_conditional_increment_sigma_diag",
    "interpolate_exact_bridges",
    "interpolate_exact_bridges_irregular_markov_sigma",
    "estimate_simulate_path_memory",
]
