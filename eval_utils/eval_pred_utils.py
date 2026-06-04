import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MultiHorizonPredictor(nn.Module):
    """GRU predictor for multi-horizon full-vector forecasting."""

    def __init__(self, data_dim, hidden_dim, max_horizon):
        super().__init__()
        self.data_dim = data_dim
        self.max_horizon = max_horizon

        self.rnn = nn.GRU(
            input_size=data_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, data_dim * max_horizon)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape: (batch, seq_len, data_dim)

        Returns
        -------
        torch.Tensor
            Shape: (batch, seq_len, max_horizon, data_dim)
        """
        h, _ = self.rnn(x)
        y = self.fc(h)

        batch, seq_len, _ = y.shape
        y = y.view(batch, seq_len, self.max_horizon, self.data_dim)

        return y


def standardize_data(data, eps=1e-8):
    mean = np.mean(data, axis=(0, 1), keepdims=True)
    std = np.std(data, axis=(0, 1), keepdims=True)
    return (data - mean) / (std + eps)


def normalize_data(data, min_val=None, max_val=None, eps=1e-8):
    if min_val is None:
        min_val = np.min(data, axis=(0, 1), keepdims=True)
    if max_val is None:
        max_val = np.max(data, axis=(0, 1), keepdims=True)

    return (data - min_val) / (max_val - min_val + eps)


def _make_multihorizon_targets(data, max_horizon):
    """
    Construct multi-horizon targets.

    Input:
        data shape: (N, T, D)

    Output:
        X shape: (N, T - max_horizon, D)
        Y shape: (N, T - max_horizon, max_horizon, D)

    For each time t:
        X[:, t, :] predicts
        Y[:, t, 0, :] = X_{t+1}
        Y[:, t, 1, :] = X_{t+2}
        ...
        Y[:, t, H-1, :] = X_{t+H}
    """
    seq_len = data.shape[1]

    X = data[:, :seq_len - max_horizon, :]

    Y = torch.stack(
        [
            data[:, h:seq_len - max_horizon + h, :]
            for h in range(1, max_horizon + 1)
        ],
        dim=2
    )

    return X, Y


def multihorizon_predictive_score_torch(
    original_data,
    synthetic_data,
    horizons=(1, 3, 5, 10),
    iterations=5000,
    scaler="standardize",
    train_min=None,
    train_max=None,
    batch_size=128,
    device=torch.device("cpu"),
    verbose=False,
    return_per_feature=True
):
    """
    Compute multi-horizon predictive score.

    The model is trained on synthetic data and evaluated on real data.

    Parameters
    ----------
    original_data : np.ndarray
        Real data of shape (N, T, D).

    synthetic_data : np.ndarray
        Synthetic data of shape (N, T, D).

    horizons : tuple[int]
        Forecast horizons to report, e.g. (1, 3, 5, 10).

    iterations : int
        Number of training iterations.

    scaler : {"standardize", "normalize", None}
        Scaling method before training/evaluation.

    train_min, train_max : np.ndarray | None
        Optional min/max values for normalization.

    batch_size : int
        Batch size.

    device : torch.device
        CPU or CUDA device.

    verbose : bool
        Print loss during training.

    return_per_feature : bool
        If True, also return MAE per horizon per feature.

    Returns
    -------
    dict
        Dictionary containing mean MAE and MAE per horizon.
    """

    assert scaler in ["standardize", "normalize", None]

    original_data = np.asarray(original_data, dtype=np.float32)
    synthetic_data = np.asarray(synthetic_data, dtype=np.float32)

    if original_data.ndim != 3 or synthetic_data.ndim != 3:
        raise ValueError("Both original_data and synthetic_data must have shape (N, T, D).")

    if original_data.shape[1:] != synthetic_data.shape[1:]:
        raise ValueError(
            f"Shape mismatch: original_data has {original_data.shape[1:]}, "
            f"synthetic_data has {synthetic_data.shape[1:]}."
        )

    horizons = tuple(sorted(set(horizons)))
    max_horizon = max(horizons)

    if original_data.shape[1] <= max_horizon:
        raise ValueError(
            f"Sequence length must be greater than max_horizon. "
            f"Got T={original_data.shape[1]} and max_horizon={max_horizon}."
        )

    # Match sample sizes
    num = min(len(original_data), len(synthetic_data))

    real_idx = np.random.permutation(len(original_data))[:num]
    synth_idx = np.random.permutation(len(synthetic_data))[:num]

    original_data = original_data[real_idx]
    synthetic_data = synthetic_data[synth_idx]

    if scaler == "standardize":
        original_data = standardize_data(original_data)
        synthetic_data = standardize_data(synthetic_data)

    elif scaler == "normalize":
        original_data = normalize_data(original_data)
        synthetic_data = normalize_data(
            synthetic_data,
            min_val=train_min,
            max_val=train_max
        )

    original_data = torch.as_tensor(original_data, dtype=torch.float32, device=device)
    synthetic_data = torch.as_tensor(synthetic_data, dtype=torch.float32, device=device)

    _, seq_len, data_dim = original_data.shape
    hidden_dim = max(data_dim // 2, 1)
    hidden_dim = 8#64#8
    print(hidden_dim)

    model = MultiHorizonPredictor(
        data_dim=data_dim,
        hidden_dim=hidden_dim,
        max_horizon=max_horizon
    ).to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.L1Loss()

    batch_size = min(batch_size, len(synthetic_data))

    X_synth, Y_synth = _make_multihorizon_targets(synthetic_data, max_horizon)
    X_real, Y_real = _make_multihorizon_targets(original_data, max_horizon)

    for itt in range(iterations):
        model.train()

        idx = torch.randperm(len(X_synth), device=device)[:batch_size]

        X_train = X_synth[idx]
        Y_train = Y_synth[idx]

        optimizer.zero_grad()

        Y_pred = model(X_train)
        loss = criterion(Y_pred, Y_train)

        loss.backward()
        optimizer.step()

        if verbose and (itt + 1) % 500 == 0:
            print(f"Iteration {itt + 1}/{iterations}, loss = {loss.item():.6f}")

    model.eval()

    with torch.no_grad():
        Y_pred_real = model(X_real)

        abs_error = torch.abs(Y_real - Y_pred_real)

        # Shape: (max_horizon,)
        mae_all_horizons = abs_error.mean(dim=(0, 1, 3))

        # Shape: (max_horizon, data_dim)
        mae_per_horizon_feature = abs_error.mean(dim=(0, 1))

    mae_by_horizon = {
        h: mae_all_horizons[h - 1].item()
        for h in horizons
    }

    result = {
        "mean_mae": float(np.mean(list(mae_by_horizon.values()))),
        "mae_by_horizon": mae_by_horizon
    }

    if return_per_feature:
        result["mae_per_horizon_feature"] = {
            h: mae_per_horizon_feature[h - 1].detach().cpu().numpy()
            for h in horizons
        }

    return result


def compute_acf(data, max_lag=20, eps=1e-8):
    """
    Compute average autocorrelation function per feature.

    Parameters
    ----------
    data : np.ndarray
        Shape: (N, T, D)

    max_lag : int
        Maximum lag.

    Returns
    -------
    np.ndarray
        ACF array of shape (max_lag, D).
        Entry [lag-1, d] is the average ACF at that lag for feature d.
    """

    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 3:
        raise ValueError("data must have shape (N, T, D).")

    N, T, D = data.shape

    if max_lag >= T:
        raise ValueError(f"max_lag must be smaller than sequence length T={T}.")

    # Center each sequence independently
    centered = data - np.mean(data, axis=1, keepdims=True)

    denom = np.sum(centered ** 2, axis=1) + eps  # (N, D)

    acfs = []

    for lag in range(1, max_lag + 1):
        numerator = np.sum(
            centered[:, :-lag, :] * centered[:, lag:, :],
            axis=1
        )  # (N, D)

        acf_lag = numerator / denom  # (N, D)
        acfs.append(np.mean(acf_lag, axis=0))  # (D,)

    return np.stack(acfs, axis=0)  # (max_lag, D)


def acf_error(original_data, synthetic_data, max_lag=20, return_details=True):
    """
    Compute ACF error between original and synthetic data.

    Returns mean absolute difference between real and synthetic ACFs.
    """

    real_acf = compute_acf(original_data, max_lag=max_lag)
    synth_acf = compute_acf(synthetic_data, max_lag=max_lag)

    error_matrix = np.abs(real_acf - synth_acf)

    result = {
        "acf_error": float(np.mean(error_matrix)),
        "acf_error_by_lag": np.mean(error_matrix, axis=1),
        "acf_error_by_feature": np.mean(error_matrix, axis=0)
    }

    if return_details:
        result["real_acf"] = real_acf
        result["synthetic_acf"] = synth_acf
        result["acf_abs_error"] = error_matrix

    return result


def compute_lagged_cross_correlation(data, max_lag=0, eps=1e-8):
    """
    Compute lagged feature cross-correlation matrices.

    For lag l, computes correlation between:

        x_t and x_{t+l}

    Parameters
    ----------
    data : np.ndarray
        Shape: (N, T, D)

    max_lag : int
        Maximum lag. Use 0 for standard feature correlation.

    Returns
    -------
    np.ndarray
        Shape: (max_lag + 1, D, D)

        result[l, i, j] = corr(feature_i at t, feature_j at t+l)
    """

    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 3:
        raise ValueError("data must have shape (N, T, D).")

    N, T, D = data.shape

    if max_lag >= T:
        raise ValueError(f"max_lag must be smaller than sequence length T={T}.")

    corr_matrices = []

    for lag in range(max_lag + 1):
        if lag == 0:
            x_left = data
            x_right = data
        else:
            x_left = data[:, :-lag, :]
            x_right = data[:, lag:, :]

        # Flatten sequences and time
        x_left = x_left.reshape(-1, D)
        x_right = x_right.reshape(-1, D)

        x_left = x_left - np.mean(x_left, axis=0, keepdims=True)
        x_right = x_right - np.mean(x_right, axis=0, keepdims=True)

        left_std = np.std(x_left, axis=0, keepdims=True) + eps
        right_std = np.std(x_right, axis=0, keepdims=True) + eps

        x_left = x_left / left_std
        x_right = x_right / right_std

        corr = (x_left.T @ x_right) / max(len(x_left) - 1, 1)

        corr_matrices.append(corr)

    return np.stack(corr_matrices, axis=0)  # (max_lag + 1, D, D)


def cross_correlation_error(
    original_data,
    synthetic_data,
    max_lag=0,
    exclude_diagonal=True,
    return_details=True
):
    """
    Compute cross-correlation error between original and synthetic data.

    Parameters
    ----------
    original_data : np.ndarray
        Real data of shape (N, T, D).

    synthetic_data : np.ndarray
        Synthetic data of shape (N, T, D).

    max_lag : int
        Maximum lag. Use 0 for normal feature-correlation matrix error.

    exclude_diagonal : bool
        If True, excludes self-correlation terms.

    Returns
    -------
    dict
        Cross-correlation errors.
    """

    real_corr = compute_lagged_cross_correlation(original_data, max_lag=max_lag)
    synth_corr = compute_lagged_cross_correlation(synthetic_data, max_lag=max_lag)

    abs_error = np.abs(real_corr - synth_corr)

    D = real_corr.shape[-1]

    if exclude_diagonal:
        if D == 1:
            mean_error = np.nan
            error_by_lag = np.full(max_lag + 1, np.nan)
        else:
            mask = ~np.eye(D, dtype=bool)
            selected_errors = abs_error[:, mask]  # (max_lag + 1, D * (D - 1))

            mean_error = float(np.mean(selected_errors))
            error_by_lag = np.mean(selected_errors, axis=1)
    else:
        mean_error = float(np.mean(abs_error))
        error_by_lag = np.mean(abs_error, axis=(1, 2))

    result = {
        "cross_corr_error": mean_error,
        "cross_corr_error_by_lag": error_by_lag
    }

    if return_details:
        result["real_cross_corr"] = real_corr
        result["synthetic_cross_corr"] = synth_corr
        result["cross_corr_abs_error"] = abs_error

    return result


def extended_time_series_metrics(
    X_data,
    X_syn,
    horizons=(1, 3, 5, 10),
    predictive_iterations=2000,
    max_acf_lag=20,
    max_cross_corr_lag=0,
    scaler="standardize",
    device=torch.device("cpu")
):
    """
    Compute extended synthetic time-series metrics.

    Returns
    -------
    dict
        Multi-horizon predictive score, ACF error, and cross-correlation error.
    """

    mh_pred = multihorizon_predictive_score_torch(
        original_data=X_data,
        synthetic_data=X_syn,
        horizons=horizons,
        iterations=predictive_iterations,
        scaler=scaler,
        device=device
    )

    acf_res = acf_error(
        original_data=X_data,
        synthetic_data=X_syn,
        max_lag=max_acf_lag,
        return_details=False
    )

    corr_res = cross_correlation_error(
        original_data=X_data,
        synthetic_data=X_syn,
        max_lag=max_cross_corr_lag,
        exclude_diagonal=True,
        return_details=False
    )

    return {
        "multi_horizon_predictive": mh_pred,
        "acf_error": acf_res,
        "cross_correlation_error": corr_res
    }

    



# -----------------------------------------------------------------------------
# TSTR / TRTR predictive baseline utilities
# -----------------------------------------------------------------------------

import random


def set_global_seed(seed, deterministic=True):
    """Seed Python, NumPy, and PyTorch for repeatable predictive runs."""
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
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        except Exception:
            pass


def _scale_train_test_data(train_data, test_data, scaler="standardize", train_min=None, train_max=None, eps=1e-8):
    """
    Scale train and test data using train-set statistics only.

    This avoids leakage from the real test set into the fitted predictor.
    """
    if scaler is None:
        return train_data, test_data

    if scaler == "standardize":
        mean = np.mean(train_data, axis=(0, 1), keepdims=True)
        std = np.std(train_data, axis=(0, 1), keepdims=True)
        return (train_data - mean) / (std + eps), (test_data - mean) / (std + eps)

    if scaler == "normalize":
        if train_min is None:
            train_min = np.min(train_data, axis=(0, 1), keepdims=True)
        if train_max is None:
            train_max = np.max(train_data, axis=(0, 1), keepdims=True)
        return (
            (train_data - train_min) / (train_max - train_min + eps),
            (test_data - train_min) / (train_max - train_min + eps),
        )

    raise ValueError("scaler must be one of {'standardize', 'normalize', None}.")


def _make_predictive_train_test_split(original_data, synthetic_data, test_ratio=0.2, seed=None):
    """
    Create aligned train/test splits for TSTR and TRTR.

    Returns
    -------
    real_train, real_test, synth_train : np.ndarray
        TRTR trains on real_train and tests on real_test.
        TSTR trains on synth_train and tests on the same real_test.
    """
    rng = np.random.default_rng(seed)

    original_data = np.asarray(original_data, dtype=np.float32)
    synthetic_data = np.asarray(synthetic_data, dtype=np.float32)

    if original_data.ndim != 3 or synthetic_data.ndim != 3:
        raise ValueError("Both original_data and synthetic_data must have shape (N, T, D).")

    if original_data.shape[1:] != synthetic_data.shape[1:]:
        raise ValueError(
            f"Shape mismatch: original_data has {original_data.shape[1:]}, "
            f"synthetic_data has {synthetic_data.shape[1:]}."
        )

    n = min(len(original_data), len(synthetic_data))
    if n < 4:
        raise ValueError("Need at least 4 real and 4 synthetic sequences.")

    real_idx = rng.permutation(len(original_data))[:n]
    synth_idx = rng.permutation(len(synthetic_data))[:n]

    real = original_data[real_idx]
    synth = synthetic_data[synth_idx]

    n_test = max(1, int(test_ratio * n))
    n_train = n - n_test

    if n_train < 1:
        raise ValueError("Train split is empty. Reduce test_ratio.")

    real_train = real[:n_train]
    real_test = real[n_train:]
    synth_train = synth[:n_train]

    return real_train, real_test, synth_train


def multihorizon_predictive_train_test_score_torch(
    train_data,
    test_data,
    horizons=(1, 3, 5, 10),
    iterations=5000,
    scaler="standardize",
    train_min=None,
    train_max=None,
    batch_size=128,
    hidden_dim=None,
    lr=1e-3,
    device=torch.device("cpu"),
    seed=None,
    verbose=False,
    return_per_feature=True,
    metric="mae",
):
    """
    Train a multi-horizon predictor on train_data and evaluate it on test_data.

    This is the shared primitive for both:
        TRTR: train_data = real_train,      test_data = real_test
        TSTR: train_data = synthetic_train, test_data = real_test

    metric : {"mae", "mse", "rmse", "huber"}
        Controls both the training loss and the evaluation aggregation.
    """
    assert scaler in ["standardize", "normalize", None]

    set_global_seed(seed)

    train_data = np.asarray(train_data, dtype=np.float32)
    test_data = np.asarray(test_data, dtype=np.float32)

    if train_data.ndim != 3 or test_data.ndim != 3:
        raise ValueError("Both train_data and test_data must have shape (N, T, D).")

    if train_data.shape[1:] != test_data.shape[1:]:
        raise ValueError(
            f"Shape mismatch: train_data has {train_data.shape[1:]}, "
            f"test_data has {test_data.shape[1:]}.")

    horizons = tuple(sorted(set(horizons)))
    max_horizon = max(horizons)

    if train_data.shape[1] <= max_horizon:
        raise ValueError(
            f"Sequence length must be greater than max_horizon. "
            f"Got T={train_data.shape[1]} and max_horizon={max_horizon}."
        )

    train_data, test_data = _scale_train_test_data(
        train_data,
        test_data,
        scaler=scaler,
        train_min=train_min,
        train_max=train_max,
    )

    train_data = torch.as_tensor(train_data, dtype=torch.float32, device=device)
    test_data = torch.as_tensor(test_data, dtype=torch.float32, device=device)

    _, _, data_dim = train_data.shape

    if hidden_dim is None:
        hidden_dim = max(data_dim // 2, 1)
        hidden_dim = 8#64#8
        print(hidden_dim)

    model = MultiHorizonPredictor(
        data_dim=data_dim,
        hidden_dim=hidden_dim,
        max_horizon=max_horizon,
    ).to(device)

    criterion, eval_fn = _build_criterion_and_eval_fn(metric)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_all, Y_train_all = _make_multihorizon_targets(train_data, max_horizon)
    X_test, Y_test = _make_multihorizon_targets(test_data, max_horizon)

    batch_size = min(batch_size, len(X_train_all))

    torch_generator = None
    if seed is not None:
        torch_generator = torch.Generator(device=device)
        torch_generator.manual_seed(seed)

    for itt in range(iterations):
        model.train()

        idx = torch.randperm(
            len(X_train_all),
            device=device,
            generator=torch_generator,
        )[:batch_size]

        X_train = X_train_all[idx]
        Y_train = Y_train_all[idx]

        optimizer.zero_grad()
        Y_pred = model(X_train)
        loss = criterion(Y_pred, Y_train)
        loss.backward()
        optimizer.step()

        if verbose and (itt + 1) % 500 == 0:
            print(f"Iteration {itt + 1}/{iterations}, loss = {loss.item():.6f}")

    model.eval()

    with torch.no_grad():
        Y_pred_test = model(X_test)
        errors = eval_fn(Y_test, Y_pred_test)                      # (N, T, max_h, D)

        score_all_horizons = _aggregate_errors(errors, metric)     # (max_h,)
        score_per_horizon_feature = errors.mean(dim=(0, 1))        # (max_h, D)
        if metric == "rmse":
            score_per_horizon_feature = torch.sqrt(score_per_horizon_feature)

    score_by_horizon = {
        h: score_all_horizons[h - 1].item()
        for h in horizons
    }

    result = {
        "mean_mae": float(np.mean(list(score_by_horizon.values()))),
        "mae_by_horizon": score_by_horizon,
    }

    if return_per_feature:
        result["mae_per_horizon_feature"] = {
            h: score_per_horizon_feature[h - 1].detach().cpu().numpy()
            for h in horizons
        }

    return result


def _build_criterion_and_eval_fn(metric: str):
    """Return (training_criterion, element_wise_error_fn) for *metric*.

    Supported values
    ----------------
    "mae"   - L1 loss / absolute error  (default)
    "mse"   - squared error
    "rmse"  - squared error (training); sqrt(mean(sq)) at aggregation
    "huber" - smooth-L1 (Huber) loss and element-wise Huber error
    """
    if metric == "mae":
        return nn.L1Loss(), lambda y, yhat: torch.abs(y - yhat)
    if metric in ("mse", "rmse"):
        return nn.MSELoss(), lambda y, yhat: (y - yhat) ** 2
    if metric == "huber":
        crit = nn.SmoothL1Loss()
        return crit, lambda y, yhat: nn.functional.smooth_l1_loss(
            yhat, y, reduction="none"
        )
    raise ValueError(
        f"Unknown metric '{metric}'. Choose from: mae, mse, rmse, huber."
    )


def _aggregate_errors(errors: torch.Tensor, metric: str) -> torch.Tensor:
    """Reduce element-wise errors (N, T, max_h, D) → per-horizon scores (max_h,)."""
    per_horizon = errors.mean(dim=(0, 1, 3))
    if metric == "rmse":
        per_horizon = torch.sqrt(per_horizon)
    return per_horizon


def _persistence_score(
    X: torch.Tensor,
    Y: torch.Tensor,
    horizons: tuple,
    metric: str = "mae",
) -> dict:
    """
    Persistence baseline from pre-built multihorizon targets.

    At each position t, predicts X[t+h] = X[t] for every horizon h.

    Parameters
    ----------
    X : (N, T-max_h, D) — scaled input sequences
    Y : (N, T-max_h, max_h, D) — scaled multi-step targets
    horizons : subset of step indices to report (1-indexed)
    metric : one of "mae", "mse", "rmse", "huber"
    """
    _, eval_fn = _build_criterion_and_eval_fn(metric)
    with torch.no_grad():
        errors = eval_fn(Y, X[:, :, None, :])           # (N, T-max_h, max_h, D)
        score_all = _aggregate_errors(errors, metric)    # (max_h,)
    score_by_horizon = {h: score_all[h - 1].item() for h in horizons}
    return {
        "mean_mae": float(np.mean(list(score_by_horizon.values()))),
        "mae_by_horizon": score_by_horizon,
    }


def multihorizon_tstr_trtr_score_torch(
    original_data,
    synthetic_data,
    horizons=(1, 3, 5, 10),
    iterations=5000,
    scaler="standardize",
    train_min=None,
    train_max=None,
    batch_size=128,
    hidden_dim=None,
    lr=1e-3,
    test_ratio=0.2,
    device=torch.device("cpu"),
    seed=None,
    verbose=False,
    return_per_feature=True,
    eps=1e-8,
    metric="mae",
):
    """
    Compute TSTR, TRTR, and persistence-baseline multi-horizon predictive scores.

    TSTR    : train on synthetic, test on real.
    TRTR    : train on real, test on held-out real.
    PERSIST : predict x_{t+h} = x_t at every horizon (no training).

    All three are evaluated in their own scaled space so skill scores are
    directly meaningful even when synthetic and real data have different scales.

    Skill scores (1 - score / persist_score):
      > 0  beats persistence          (good)
      ≈ 0  equivalent to persistence  (degenerate predictor)
      < 0  worse than persistence     (very bad)

    metric : {"mae", "mse", "rmse", "huber"}
        Controls both the training loss and the evaluation aggregation.
    """
    real_train, real_test, synth_train = _make_predictive_train_test_split(
        original_data,
        synthetic_data,
        test_ratio=test_ratio,
        seed=seed,
    )

    trtr = multihorizon_predictive_train_test_score_torch(
        train_data=real_train,
        test_data=real_test,
        horizons=horizons,
        iterations=iterations,
        scaler=scaler,
        train_min=train_min,
        train_max=train_max,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        lr=lr,
        device=device,
        seed=seed,
        verbose=verbose,
        return_per_feature=return_per_feature,
        metric=metric,
    )

    tstr = multihorizon_predictive_train_test_score_torch(
        train_data=synth_train,
        test_data=real_test,
        horizons=horizons,
        iterations=iterations,
        scaler=scaler,
        train_min=train_min,
        train_max=train_max,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        lr=lr,
        device=device,
        seed=seed,
        verbose=verbose,
        return_per_feature=return_per_feature,
        metric=metric,
    )

    horizons = tuple(sorted(set(horizons)))

    # TRTR persistence: real_test scaled by real_train stats — same space as TRTR score
    _, real_test_trtr_scaled = _scale_train_test_data(
        real_train, real_test, scaler=scaler,
        train_min=train_min, train_max=train_max,
    )
    real_test_trtr_t = torch.as_tensor(
        np.asarray(real_test_trtr_scaled, dtype=np.float32), dtype=torch.float32, device=device
    )
    X_trtr_p, Y_trtr_p = _make_multihorizon_targets(real_test_trtr_t, max(horizons))
    persist_trtr = _persistence_score(X_trtr_p, Y_trtr_p, horizons, metric=metric)

    # TSTR persistence: real_test scaled by synth_train stats — same space as TSTR score
    _, real_test_tstr_scaled = _scale_train_test_data(
        synth_train, real_test, scaler=scaler,
        train_min=train_min, train_max=train_max,
    )
    real_test_tstr_t = torch.as_tensor(
        np.asarray(real_test_tstr_scaled, dtype=np.float32), dtype=torch.float32, device=device
    )
    X_tstr_p, Y_tstr_p = _make_multihorizon_targets(real_test_tstr_t, max(horizons))
    persist_tstr = _persistence_score(X_tstr_p, Y_tstr_p, horizons, metric=metric)

    gap_by_horizon = {
        h: float(tstr["mae_by_horizon"][h] - trtr["mae_by_horizon"][h])
        for h in horizons
    }
    ratio_by_horizon = {
        h: float(tstr["mae_by_horizon"][h] / (trtr["mae_by_horizon"][h] + eps))
        for h in horizons
    }
    tstr_skill_by_horizon = {
        h: float(1.0 - tstr["mae_by_horizon"][h] / (persist_tstr["mae_by_horizon"][h] + eps))
        for h in horizons
    }
    trtr_skill_by_horizon = {
        h: float(1.0 - trtr["mae_by_horizon"][h] / (persist_trtr["mae_by_horizon"][h] + eps))
        for h in horizons
    }

    return {
        "TSTR": tstr,
        "TRTR": trtr,
        "PERSIST": persist_trtr,
        "PERSIST_TSTR": persist_tstr,
        "gap_mean_mae":          float(tstr["mean_mae"] - trtr["mean_mae"]),
        "ratio_mean_mae":        float(tstr["mean_mae"] / (trtr["mean_mae"] + eps)),
        "gap_by_horizon":        gap_by_horizon,
        "ratio_by_horizon":      ratio_by_horizon,
        "tstr_skill_mean":       float(1.0 - tstr["mean_mae"] / (persist_tstr["mean_mae"] + eps)),
        "trtr_skill_mean":       float(1.0 - trtr["mean_mae"] / (persist_trtr["mean_mae"] + eps)),
        "tstr_skill_by_horizon": tstr_skill_by_horizon,
        "trtr_skill_by_horizon": trtr_skill_by_horizon,
    }


# -----------------------------------------------------------------------------
# Limited-data three-regime evaluation
# -----------------------------------------------------------------------------

def three_regime_predictive_score(
    X_train_real: np.ndarray,
    X_train_syn: np.ndarray,
    X_test: np.ndarray,
    horizons=(1, 3, 5, 10),
    iterations=5000,
    scaler=None,
    batch_size=128,
    hidden_dim=None,
    lr=1e-3,
    device=torch.device("cpu"),
    seed=None,
    verbose=False,
    return_per_feature=False,
    metric="rmse",
) -> dict:
    """
    Evaluate a downstream GRU forecaster under three training regimes.

    All three predictors are evaluated on the same *X_test* (held-out real),
    matching the limited-data experiment protocol from the paper.

    Regimes
    -------
    real_only : train on X_train_real  (Real_r)
    syn_only  : train on X_train_syn   (Synthetic_r)
    mixed     : train on Real_r u Synthetic_r

    Parameters
    ----------
    X_train_real : np.ndarray, shape (N_r, T, D)
        Subsampled real training set Real_r.
    X_train_syn : np.ndarray, shape (N_s, T, D)
        Synthetic samples generated from Real_r. Typically N_s == N_r.
    X_test : np.ndarray, shape (N_test, T, D)
        Held-out real test set — same split used in the main experiments.
    horizons : tuple[int]
        Forecast horizons to evaluate.
    iterations : int
        GRU training iterations for each regime.
    scaler : {"standardize", "normalize", None}
        Scaling applied independently per regime using its own train-set
        statistics. Default None evaluates RMSE in raw data space.
    batch_size : int
    hidden_dim : int | None
    lr : float
    device : torch.device
    seed : int | None
        Passed to every predictor for reproducibility.
    verbose : bool
    return_per_feature : bool
    metric : {"mae", "mse", "rmse", "huber"}

    Returns
    -------
    dict
        Keys ``"real_only"``, ``"syn_only"``, ``"mixed"``, each containing
        the result dict from ``multihorizon_predictive_train_test_score_torch``.
    """
    X_train_real = np.asarray(X_train_real, dtype=np.float32)
    X_train_syn  = np.asarray(X_train_syn,  dtype=np.float32)
    X_test       = np.asarray(X_test,       dtype=np.float32)

    if X_train_real.shape[1:] != X_test.shape[1:]:
        raise ValueError(
            f"X_train_real shape {X_train_real.shape[1:]} does not match "
            f"X_test shape {X_test.shape[1:]}."
        )
    if X_train_syn.shape[1:] != X_test.shape[1:]:
        raise ValueError(
            f"X_train_syn shape {X_train_syn.shape[1:]} does not match "
            f"X_test shape {X_test.shape[1:]}."
        )

    common_kwargs = dict(
        horizons=horizons,
        iterations=iterations,
        scaler=scaler,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        lr=lr,
        device=device,
        seed=seed,
        verbose=verbose,
        return_per_feature=return_per_feature,
        metric=metric,
    )

    X_mixed = np.concatenate([X_train_real, X_train_syn], axis=0)

    print(f"[three_regime] real={X_train_real.shape[0]}  syn={X_train_syn.shape[0]}  "
          f"mixed={X_mixed.shape[0]}  test={X_test.shape[0]}")

    real_only = multihorizon_predictive_train_test_score_torch(
        train_data=X_train_real, test_data=X_test, **common_kwargs
    )
    syn_only = multihorizon_predictive_train_test_score_torch(
        train_data=X_train_syn, test_data=X_test, **common_kwargs
    )
    mixed = multihorizon_predictive_train_test_score_torch(
        train_data=X_mixed, test_data=X_test, **common_kwargs
    )

    return {
        "real_only": real_only,
        "syn_only":  syn_only,
        "mixed":     mixed,
    }
