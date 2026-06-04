import numpy as np
import numba as nb
from tqdm import tqdm


NEG_INF = -1e300

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
def kernel_value_from_sqnorm(sq_norm, h, kernel_type):
    hh = h * h

    if kernel_type == KERNEL_QUARTIC:
        if sq_norm >= hh:
            return 0.0
        val = hh - sq_norm
        return val * val

    elif kernel_type == KERNEL_GAUSSIAN:
        return np.exp(-0.5 * sq_norm / hh)

    elif kernel_type == KERNEL_EPANECHNIKOV:
        if sq_norm >= hh:
            return 0.0
        return 1.0 - sq_norm / hh

    elif kernel_type == KERNEL_TRIANGULAR:
        norm_u = np.sqrt(sq_norm)
        if norm_u >= h:
            return 0.0
        return 1.0 - norm_u / h

    if sq_norm >= hh:
        return 0.0
    val = hh - sq_norm
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
def truncated_history_weights(i, K, X, path, h, kernel_type):
    """
    Compute normalized kernel weights for transition [t_i, t_{i+1}].

    Returns
    -------
    weights : np.ndarray, shape (M,)
        Normalized weights.
    hist_len_used : int
        Effective history length used.
    """
    M = X.shape[0]
    d = X.shape[2]

    max_hist = K
    if i + 1 < max_hist:
        max_hist = i + 1

    logw = np.empty(M, dtype=np.float64)

    for hist_len in range(max_hist, 0, -1):
        start = i - hist_len + 1
        max_logw = NEG_INF
        active = 0

        for m in range(M):
            acc = 0.0
            alive = True

            for j in range(start, i + 1):
                sq_norm = 0.0
                for r in range(d):
                    diff = path[j, r] - X[m, j, r]
                    sq_norm += diff * diff

                kval = kernel_value_from_sqnorm(sq_norm, h, kernel_type)

                if kval <= 0.0:
                    alive = False
                    break

                acc += np.log(kval)

            if alive:
                logw[m] = acc
                active += 1
                if acc > max_logw:
                    max_logw = acc
            else:
                logw[m] = NEG_INF

        if active > 0:
            weights = np.zeros(M, dtype=np.float64)
            total = 0.0

            for m in range(M):
                if logw[m] > NEG_INF / 2:
                    w = np.exp(logw[m] - max_logw)
                    weights[m] = w
                    total += w

            if total > 0.0:
                for m in range(M):
                    weights[m] /= total

                return weights, hist_len

    # Uniform fallback
    weights = np.empty(M, dtype=np.float64)
    for m in range(M):
        weights[m] = 1.0 / M

    return weights, 0


@nb.njit(cache=True)
def simulate_exact_latent_endpoint_grid_core(N, K, X, h, kernel_type, x0):
    """
    Simulate one path on the observation grid using conditional transition probabilities.

    Parameters
    ----------
    N : int
        Number of time points in the path.

    Returns
    -------
    path : np.ndarray, shape (N, d)
    chosen_labels : np.ndarray, shape (N - 1,)
    hist_used : np.ndarray, shape (N - 1,)
    """
    d = X.shape[2]
    n_transitions = N - 1

    path = np.empty((N, d), dtype=X.dtype)
    chosen_labels = np.empty(n_transitions, dtype=np.int64)
    hist_used = np.empty(n_transitions, dtype=np.int64)

    for r in range(d):
        path[0, r] = x0[r]

    for i in range(n_transitions):
        weights, hist_len = truncated_history_weights(i, K, X, path, h, kernel_type)
        m = sample_categorical(weights)

        chosen_labels[i] = m
        hist_used[i] = hist_len

        for r in range(d):
            path[i + 1, r] = X[m, i + 1, r]

    return path, chosen_labels, hist_used


def simulate_exact_latent_endpoint_grid(
    N,
    M,
    d,
    K,
    X,
    h,
    kernel_type="quartic",
    x0=None,
    sample_initial=True,
):
    """
    Parameters
    ----------
    N : int
        Number of time points in each path.

    Returns
    -------
    path : np.ndarray, shape (N, d)
    chosen_labels : np.ndarray, shape (N - 1,)
    hist_used : np.ndarray, shape (N - 1,)
    """
    if X.shape != (M, N, d):
        raise ValueError(f"Expected X.shape == ({M}, {N}, {d}), got {X.shape}.")

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
        N, K, X, h, kernel_code, x0_
    )


@nb.njit(cache=True)
def interpolate_exact_bridges(grid_path, deltati, N_pi, sigma=1.0):
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
    h,
    M_simu,
    kernel_type="quartic",
    x0=None,
    sample_initial=True,
):
    """
    Parameters
    ----------
    N : int
        Number of time points in each path.

    Returns
    -------
    data_sb : np.ndarray, shape (M_simu, N, d)
    chosen_labels : np.ndarray, shape (M_simu, N - 1)
    hist_used : np.ndarray, shape (M_simu, N - 1)
    """
    if X.shape != (M, N, d):
        raise ValueError(f"Expected X.shape == ({M}, {N}, {d}), got {X.shape}.")

    kernel_code = parse_kernel_type(kernel_type)

    data_syn = np.empty((M_simu, N, d), dtype=X.dtype)
    chosen_labels = np.empty((M_simu, N - 1), dtype=np.int64)
    hist_used = np.empty((M_simu, N - 1), dtype=np.int64)

   
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

        path, labels_k, hist_k = simulate_exact_latent_endpoint_grid_core(
            N, K, X, h, kernel_code, x0_k
        )

        data_syn[k, :, :] = path
        chosen_labels[k, :] = labels_k
        hist_used[k, :] = hist_k


    return data_syn, chosen_labels, hist_used



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

    # Persistent arrays returned by simulate_path
    data_sb_bytes = M_simu * N * d * x_itemsize
    chosen_labels_bytes = M_simu * (N - 1) * int64_size
    hist_used_bytes = M_simu * (N - 1) * int64_size

    returned_total = data_sb_bytes + chosen_labels_bytes + hist_used_bytes

    # Temporary arrays for one call of simulate_exact_latent_endpoint_grid_core
    path_bytes = N * d * x_itemsize
    labels_k_bytes = (N - 1) * int64_size
    hist_k_bytes = (N - 1) * int64_size

    # Temporary arrays inside truncated_history_weights
    # logw is always allocated; weights is also allocated before return.
    logw_bytes = M * np.dtype(np.float64).itemsize
    weights_bytes = M * np.dtype(np.float64).itemsize

    temporary_peak = (
        path_bytes
        + labels_k_bytes
        + hist_k_bytes
        + logw_bytes
        + weights_bytes
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
    print(f"  returned total:          {format_bytes(returned_total)}")
    print()
    print("Temporary peak, approx:")
    print(f"  one path:                {format_bytes(path_bytes)}")
    print(f"  labels_k:                {format_bytes(labels_k_bytes)}")
    print(f"  hist_k:                  {format_bytes(hist_k_bytes)}")
    print(f"  logw:                    {format_bytes(logw_bytes)}")
    print(f"  weights:                 {format_bytes(weights_bytes)}")
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
        "returned_total": returned_total,
        "temporary_peak": temporary_peak,
        "extra_memory": returned_total + temporary_peak,
        "total_including_X": x_bytes + returned_total + temporary_peak,
    }