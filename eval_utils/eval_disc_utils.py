import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


import random

def set_global_seed(seed):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




class GRUDiscriminator(nn.Module):
    """GRU-based real-vs-synthetic discriminator."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        h = h[-1]
        logits = self.fc(h).squeeze(-1)
        return logits


class TCNDiscriminator(nn.Module):
    """Simple temporal CNN discriminator."""

    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        x = x.transpose(1, 2)  # (batch, dim, seq_len)
        h = self.net(x).squeeze(-1)
        logits = self.fc(h).squeeze(-1)
        return logits


def _standardize_from_train(X_train, X_test, eps=1e-8):
    """
    Standardize using combined training data only.

    Important: do not standardize real and synthetic separately for
    discriminative scoring, because that can hide distributional differences.
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)

    X_train = (X_train - mean) / (std + eps)
    X_test = (X_test - mean) / (std + eps)

    return X_train, X_test


def _make_balanced_discriminative_split(
    original_data,
    synthetic_data,
    test_ratio=0.2,
    seed=None
):
    """
    Create balanced train/test splits for real-vs-synthetic classification.
    """

    rng = np.random.default_rng(seed)

    original_data = np.asarray(original_data, dtype=np.float32)
    synthetic_data = np.asarray(synthetic_data, dtype=np.float32)

    if original_data.ndim != 3 or synthetic_data.ndim != 3:
        raise ValueError("Both inputs must have shape (N, T, D).")

    if original_data.shape[1:] != synthetic_data.shape[1:]:
        raise ValueError(
            f"Shape mismatch: original has {original_data.shape[1:]}, "
            f"synthetic has {synthetic_data.shape[1:]}."
        )

    n = min(len(original_data), len(synthetic_data))

    if n < 4:
        raise ValueError("Need at least 4 samples from each dataset.")

    real_idx = rng.permutation(len(original_data))[:n]
    synth_idx = rng.permutation(len(synthetic_data))[:n]

    real = original_data[real_idx]
    synth = synthetic_data[synth_idx]

    n_test = max(1, int(test_ratio * n))
    n_train = n - n_test

    if n_train < 1:
        raise ValueError("Train split is empty. Reduce test_ratio.")

    real_train, real_test = real[:n_train], real[n_train:]
    synth_train, synth_test = synth[:n_train], synth[n_train:]

    X_train = np.concatenate([real_train, synth_train], axis=0)
    y_train = np.concatenate([
        np.ones(len(real_train)),
        np.zeros(len(synth_train))
    ])

    X_test = np.concatenate([real_test, synth_test], axis=0)
    y_test = np.concatenate([
        np.ones(len(real_test)),
        np.zeros(len(synth_test))
    ])

    train_perm = rng.permutation(len(X_train))
    test_perm = rng.permutation(len(X_test))

    X_train = X_train[train_perm]
    y_train = y_train[train_perm]

    X_test = X_test[test_perm]
    y_test = y_test[test_perm]

    return X_train, y_train, X_test, y_test


def _classification_metrics(y_true, y_prob):
    """
    Return discriminative metrics.

    Lower is better for:
        discriminative_score_acc
        discriminative_score_auroc

    Random guessing gives approximately:
        accuracy = 0.5
        AUROC = 0.5
        AUPRC = positive class ratio
    """

    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = np.nan

    try:
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = np.nan

    positive_rate = np.mean(y_true)

    return {
        "accuracy": float(acc),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "positive_rate": float(positive_rate),

        # TimeGAN-style score: lower is better, ideal is 0
        "discriminative_score_acc": float(abs(acc - 0.5)),

        # AUROC-based separability: lower is better, ideal is 0
        "discriminative_score_auroc": float(abs(auroc - 0.5))
        if not np.isnan(auroc) else np.nan
    }


def neural_discriminative_score_torch(
    original_data,
    synthetic_data,
    architecture="gru",
    iterations=2000,
    batch_size=128,
    hidden_dim=None,
    test_ratio=0.2,
    standardize=True,
    lr=1e-3,
    device=torch.device("cpu"),
    seed=None,
    verbose=False
):
    """
    Extended discriminative score using a neural classifier.

    The classifier is trained to distinguish real from synthetic sequences.

    Labels:
        real = 1
        synthetic = 0

    Parameters
    ----------
    original_data : np.ndarray
        Real data, shape (N, T, D).

    synthetic_data : np.ndarray
        Synthetic data, shape (N, T, D).

    architecture : {"gru", "tcn"}
        Discriminator architecture.

    iterations : int
        Training iterations.

    batch_size : int
        Batch size.

    hidden_dim : int | None
        Hidden dimension. Defaults to max(D // 2, 1).

    test_ratio : float
        Fraction of data used for held-out evaluation.

    standardize : bool
        If True, standardize using the combined training set.

    lr : float
        Learning rate.

    device : torch.device
        CPU or CUDA device.

    seed : int | None
        Random seed.

    verbose : bool
        Print training loss.

    Returns
    -------
    dict
        Accuracy, AUROC, AUPRC, and lower-is-better discriminative scores.
    """
    set_global_seed(seed)
    X_train, y_train, X_test, y_test = _make_balanced_discriminative_split(
        original_data,
        synthetic_data,
        test_ratio=test_ratio,
        seed=seed
    )

    if standardize:
        X_train, X_test = _standardize_from_train(X_train, X_test)

    _, _, data_dim = X_train.shape

    if hidden_dim is None:
        hidden_dim = max(data_dim // 2, 1)

    if architecture == "gru":
        model = GRUDiscriminator(
            input_dim=data_dim,
            hidden_dim=hidden_dim
        ).to(device)

    elif architecture == "tcn":
        model = TCNDiscriminator(
            input_dim=data_dim,
            hidden_dim=hidden_dim
        ).to(device)

    else:
        raise ValueError("architecture must be one of {'gru', 'tcn'}.")

    X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y_train, dtype=torch.float32, device=device)

    X_test_t = torch.as_tensor(X_test, dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    batch_size = min(batch_size, len(X_train))
    torch_generator = torch.Generator(device=device)
    torch_generator.manual_seed(seed)

    for itt in range(iterations):
        model.train()

        #idx = torch.randperm(len(X_train), device=device)[:batch_size]
        idx = torch.randperm(
                                len(X_train),
                                device=device,
                                generator=torch_generator
                            )[:batch_size]


        X_mb = X_train[idx]
        y_mb = y_train[idx]

        optimizer.zero_grad()

        logits = model(X_mb)
        loss = criterion(logits, y_mb)

        loss.backward()
        optimizer.step()

        if verbose and (itt + 1) % 500 == 0:
            print(
                f"{architecture.upper()} discriminator "
                f"iteration {itt + 1}/{iterations}, loss = {loss.item():.6f}"
            )

    model.eval()

    with torch.no_grad():
        logits = model(X_test_t)
        y_prob = torch.sigmoid(logits).detach().cpu().numpy()

    metrics = _classification_metrics(y_test, y_prob)
    metrics["architecture"] = architecture

    return metrics



def extract_time_series_summary_features(data, max_acf_lag=3, eps=1e-8):
    """
    Extract sequence-level summary features.

    Features include:
        mean
        std
        min
        max
        median
        quartiles
        mean/std of first differences
        autocorrelation per feature

    Parameters
    ----------
    data : np.ndarray
        Shape (N, T, D)

    Returns
    -------
    np.ndarray
        Shape (N, num_summary_features)
    """

    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 3:
        raise ValueError("data must have shape (N, T, D).")

    N, T, D = data.shape

    features = []

    features.append(np.mean(data, axis=1))
    features.append(np.std(data, axis=1))
    features.append(np.min(data, axis=1))
    features.append(np.max(data, axis=1))
    features.append(np.median(data, axis=1))
    features.append(np.percentile(data, 25, axis=1))
    features.append(np.percentile(data, 75, axis=1))

    diffs = np.diff(data, axis=1)

    features.append(np.mean(diffs, axis=1))
    features.append(np.std(diffs, axis=1))

    centered = data - np.mean(data, axis=1, keepdims=True)
    denom = np.sum(centered ** 2, axis=1) + eps

    max_acf_lag = min(max_acf_lag, T - 1)

    for lag in range(1, max_acf_lag + 1):
        numerator = np.sum(
            centered[:, :-lag, :] * centered[:, lag:, :],
            axis=1
        )

        acf_lag = numerator / denom
        features.append(acf_lag)

    return np.concatenate(features, axis=1)


def summary_discriminative_score(
    original_data,
    synthetic_data,
    classifier="logistic",
    test_ratio=0.2,
    max_acf_lag=3,
    seed=None
):
    """
    Discriminative score using handcrafted summary features.

    Parameters
    ----------
    classifier : {"logistic", "random_forest"}
        Classifier type.

    Returns
    -------
    dict
        Accuracy, AUROC, AUPRC, and discriminative scores.
    """

    X_train, y_train, X_test, y_test = _make_balanced_discriminative_split(
        original_data,
        synthetic_data,
        test_ratio=test_ratio,
        seed=seed
    )

    F_train = extract_time_series_summary_features(
        X_train,
        max_acf_lag=max_acf_lag
    )

    F_test = extract_time_series_summary_features(
        X_test,
        max_acf_lag=max_acf_lag
    )

    if classifier == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000)
        )

    elif classifier == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight="balanced"
        )

    else:
        raise ValueError("classifier must be one of {'logistic', 'random_forest'}.")

    model.fit(F_train, y_train)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(F_test)[:, 1]
    else:
        y_prob = model.decision_function(F_test)
        y_prob = 1.0 / (1.0 + np.exp(-y_prob))

    metrics = _classification_metrics(y_test, y_prob)
    metrics["architecture"] = f"summary_{classifier}"

    return metrics


def extended_discriminative_scores(
    X_data,
    X_syn,
    iterations=2000,
    batch_size=128,
    test_ratio=0.2,
    standardize=True,
    device=torch.device("cpu"),
    seed=None
):
    """
    Compute multiple discriminative scores.

    Returns
    -------
    dict
        Results from GRU, TCN, logistic-summary, and RF-summary discriminators.
    """

    results = {}

    results["gru"] = neural_discriminative_score_torch(
        original_data=X_data,
        synthetic_data=X_syn,
        architecture="gru",
        iterations=iterations,
        batch_size=batch_size,
        test_ratio=test_ratio,
        standardize=standardize,
        device=device,
        seed=seed
    )

    results["tcn"] = neural_discriminative_score_torch(
        original_data=X_data,
        synthetic_data=X_syn,
        architecture="tcn",
        iterations=iterations,
        batch_size=batch_size,
        test_ratio=test_ratio,
        standardize=standardize,
        device=device,
        seed=seed
    )

    results["summary_logistic"] = summary_discriminative_score(
        original_data=X_data,
        synthetic_data=X_syn,
        classifier="logistic",
        test_ratio=test_ratio,
        seed=seed
    )

    results["summary_random_forest"] = summary_discriminative_score(
        original_data=X_data,
        synthetic_data=X_syn,
        classifier="random_forest",
        test_ratio=test_ratio,
        seed=seed
    )

    acc_scores = [
        results[key]["discriminative_score_acc"]
        for key in results
    ]

    auroc_scores = [
        results[key]["discriminative_score_auroc"]
        for key in results
    ]

    results["aggregate"] = {
        "mean_discriminative_score_acc": float(np.nanmean(acc_scores)),
        "mean_discriminative_score_auroc": float(np.nanmean(auroc_scores))
    }

    return results



def aggregate_discriminative_scores(results):
    sequence_keys = ["gru", "tcn"]
    summary_keys = ["summary_logistic", "summary_random_forest"]

    sequence_acc_scores = [
        results[key]["discriminative_score_acc"]
        for key in sequence_keys
    ]

    sequence_auc_scores = [
        results[key]["discriminative_score_auroc"]
        for key in sequence_keys
    ]

    summary_acc_scores = [
        results[key]["discriminative_score_acc"]
        for key in summary_keys
    ]

    summary_auc_scores = [
        results[key]["discriminative_score_auroc"]
        for key in summary_keys
    ]

    results["aggregate"] = {
        "DS_acc_GRU_TCN": float(np.nanmean(sequence_acc_scores)),
        "DS_auc_GRU_TCN": float(np.nanmean(sequence_auc_scores)),
        "DS_acc_LR_RF": float(np.nanmean(summary_acc_scores)),
        "DS_auc_LR_RF": float(np.nanmean(summary_auc_scores)),
    }

    return results['aggregate']