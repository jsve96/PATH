#!/usr/bin/env python3
"""
Visualise PATH scalability results from results/path_scalability.json.

Produces:
  1. path_scalability_main.{pdf,png}
       2×3 grid: rows = {sim_time, DS_acc}, cols = {N, T, d}
       Mean ± std across seeds; log–log fitted slope on sim_time panels.
  2. path_scalability_memory.{pdf,png}
       1×3 strip: peak_mem vs {N, T, d} (appendix / supplementary)

Usage:
  python ablation/visualize_scalability.py
  python ablation/visualize_scalability.py --results results/path_scalability.json
  python ablation/visualize_scalability.py --metric DS_auc_GRU_TCN
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ── paths ─────────────────────────────────────────────────────────────────────
DEFAULT_RESULTS = Path("results/path_scalability.json")
DEFAULT_OUT     = Path("results/img")

# ── style ─────────────────────────────────────────────────────────────────────
REF_COLOR = "#999999"           # fitted-slope dashed line


def _setup_style() -> None:
    sns.set_style("ticks")
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         9,
        "axes.labelsize":    10,
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   8,
        "legend.framealpha": 0.85,
        "legend.edgecolor":  "0.8",
        "lines.linewidth":   1.6,
        "lines.markersize":  5,
        "axes.linewidth":    0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "grid.linewidth":    0.5,
        "grid.color":        "0.88",
        "figure.dpi":        120,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
    })


# ── data loading ──────────────────────────────────────────────────────────────

def load_df(path: Path, metric: str) -> pd.DataFrame:
    with open(path) as f:
        raw = json.load(f)

    records = []
    for entry in raw.values():
        if "error" in entry:
            continue
        agg = entry.get("aggregate", {})
        records.append({
            "axis_N":       entry["axis_N"],
            "axis_T":       entry["axis_T"],
            "axis_d":       entry["axis_d"],
            "seed":         entry["seed"],
            "sim_time_s":   entry["sim_time_s"],
            "peak_mem_mb":  entry.get("peak_mem_mb", float("nan")),
            "metric":       float(agg.get(metric, float("nan"))),
        })

    if not records:
        raise ValueError(f"No valid results in {path}.")
    return pd.DataFrame(records)


_BASELINES = {"N": 1000, "T": 24, "d": 6}


def _agg(df: pd.DataFrame, axis: str, col: str) -> pd.DataFrame:
    """Aggregate over seeds, filtering to rows where all other axes are fixed
    at their OFAT baseline so cross-axis entries don't pollute the group."""
    vary_col = f"axis_{axis}"
    mask = pd.Series(True, index=df.index)
    for other, baseline in _BASELINES.items():
        if other != axis:
            mask &= df[f"axis_{other}"] == baseline
    sub = df[mask]
    grp = sub.groupby(vary_col)[col].agg(["mean", "std"]).reset_index()
    grp.columns = [vary_col, "mean", "std"]
    return grp.sort_values(vary_col)


# ── slope fitting ─────────────────────────────────────────────────────────────

def _fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Log–log OLS slope (empirical complexity exponent)."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    lx, ly = np.log10(x[mask]), np.log10(y[mask])
    slope = np.polyfit(lx, ly, 1)[0]
    return round(slope, 2)


def _draw_fitted_line(ax, x: np.ndarray, y: np.ndarray) -> None:
    """Overlay a dashed log–log fit line and annotate slope."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return
    lx = np.log10(x[mask])
    slope, intercept = np.polyfit(lx, np.log10(y[mask]), 1)
    x_fit = np.logspace(lx.min(), lx.max(), 50)
    y_fit = 10 ** (slope * np.log10(x_fit) + intercept)
    ax.plot(x_fit, y_fit, "--", color=REF_COLOR, linewidth=1.2, zorder=2)
    # annotate slope in upper-left of log–log space
    ax.text(
        0.05, 0.93, f"slope {slope:.2f}",
        transform=ax.transAxes, fontsize=7.5,
        color=REF_COLOR, va="top",
    )


# ── save helper ───────────────────────────────────────────────────────────────

def _save(fig, stem: Path) -> None:
    for ext in (".pdf", ".png"):
        p = stem.with_suffix(ext)
        fig.savefig(p)
        print(f"  Saved: {p}")
    plt.close(fig)


# ── main figure: 1 × 3 with dual y-axes (sim_time left, DS_acc right) ────────

# Left axis:  sim_time  — blue
# Right axis: DS_acc    — orange (second Wong colour)
COLOR_TIME   = "black"
COLOR_TIME_F = "#56B4E9"
COLOR_QUAL   = "#E69F00"
COLOR_QUAL_F = "#F5C842"


def plot_main(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    _setup_style()

    axes_cfg = [
        ("N", "axis_N", True,  "$N$ (training samples)"),
        ("T", "axis_T", False, "$T$ (sequence length)"),
        ("d", "axis_d", True,  "$d$ (feature dimension)"),
    ]

    # baseline DS_acc (N=1000, T=24, d=6) for reference line
    try:
        baseline_val = df[
            (df["axis_N"] == 1000) & (df["axis_T"] == 24) & (df["axis_d"] == 6)
        ]["metric"].mean()
    except Exception:
        baseline_val = float("nan")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)

    for col_idx, (axis, col, use_log_x, xlabel) in enumerate(axes_cfg):
        ax_l = axes[col_idx]
        ax_r = ax_l.twinx()

        # ── left: simulation time ──────────────────────────────────────────
        grp_t = _agg(df, axis, "sim_time_s")
        x_t   = grp_t[col].values
        y_t   = grp_t["mean"].values
        ye_t  = grp_t["std"].fillna(0).values

        l1, = ax_l.plot(x_t, y_t, "-o", color=COLOR_TIME, zorder=3,
                        label="Sim. time (s)")
        ax_l.fill_between(x_t, np.maximum(y_t - ye_t, 1e-9), y_t + ye_t,
                          alpha=0.18, color=COLOR_TIME_F, zorder=2)
        ax_l.set_yscale("log")

        if use_log_x:
            ax_l.set_xscale("log")
            _draw_fitted_line(ax_l, x_t, y_t)

        # colour left spine + ticks to match line
        #ax_l.spines["left"].set_color(COLOR_TIME)
        #ax_l.tick_params(axis="y", colors=COLOR_TIME)
        #ax_l.yaxis.label.set_color(COLOR_TIME)
        if col_idx == 0:
            ax_l.set_ylabel("Simulation time (s)", color='black')

        # ── right: DS_acc ──────────────────────────────────────────────────
        grp_q = _agg(df, axis, "metric")
        x_q   = grp_q[col].values
        y_q   = grp_q["mean"].values
        ye_q  = grp_q["std"].fillna(0).values

        l2, = ax_r.plot(x_q, y_q, "-s", color=COLOR_QUAL, zorder=3,
                        label=metric)
        ax_r.fill_between(x_q, np.maximum(y_q - ye_q, 0), y_q + ye_q,
                          alpha=0.18, color=COLOR_QUAL_F, zorder=2)

        if np.isfinite(baseline_val):
            ax_r.axhline(baseline_val, color=COLOR_QUAL, linewidth=0.9,
                         linestyle=":", zorder=1)

        ax_r.set_ylim(bottom=0)
        ax_r.spines["right"].set_color(COLOR_QUAL)
        ax_r.tick_params(axis="y", colors=COLOR_QUAL)
        ax_r.yaxis.label.set_color(COLOR_QUAL)
        if col_idx == 2:
            ax_r.set_ylabel(metric, color=COLOR_QUAL)

        # hide right spine on left/middle panels to reduce clutter
        if col_idx < 2:
            ax_r.set_yticklabels([])
            ax_r.tick_params(axis="y", length=0)

        # hide unused spines
        ax_l.spines["top"].set_visible(False)
        ax_r.spines["top"].set_visible(False)

        ax_l.set_xlabel(xlabel, labelpad=3)
        ax_l.set_title(f"Vary {axis}")
        ax_l.grid(axis="y", color="0.88", linewidth=0.5, zorder=0)

        # shared legend on first panel only
        if col_idx == 0:
            ax_l.legend(
                handles=[l1, l2], loc="upper left",
                fontsize=7.5, framealpha=0.85,
            )

    fig.suptitle(
        f"PATH scalability — KDD Cup  |  metric: {metric}  "
        r"(lower DS_acc $\rightarrow$ more realistic)",
        fontsize=9.5, y=1.02, color="#222222",
    )

    _save(fig, out_dir / f"path_scalability_main_{metric}")

# ── runtime × memory figure: 1 × 3 dual-axis ─────────────────────────────────

# Right axis: peak memory — green (third Wong colour)
COLOR_MEM   = "grey"
COLOR_MEM_F = "#66C2A5"

import matplotlib.ticker as mticker

def plot_runtime_memory(df: pd.DataFrame, out_dir: Path) -> None:
    _setup_style()

    def _plain_tick(y, _):
        return f"{y:.2f}".rstrip("0").rstrip(".")

    def _set_plain_log_ticks(ax, values, n_ticks=5):
        values = np.asarray(values)
        values = values[np.isfinite(values) & (values > 0)]

        if len(values) == 0:
            return

        ymin, ymax = values.min(), values.max()

        pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.05 * ymax
        ymin = max(ymin - pad, 1e-12)
        ymax = ymax + pad

        ticks = np.linspace(ymin, ymax, n_ticks)

        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_plain_tick))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.yaxis.get_offset_text().set_visible(False)

    axes_cfg = [
        ("N", "axis_N", True,  "$N$ (training samples)"),
        ("T", "axis_T", False, "$T$ (sequence length)"),
        ("d", "axis_d", True,  "$d$ (feature dimension)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)

    for col_idx, (axis, col, use_log_x, xlabel) in enumerate(axes_cfg):
        ax_l = axes[col_idx]
        ax_r = ax_l.twinx()

        grp_t = _agg(df, axis, "sim_time_s")
        x_t = grp_t[col].values
        y_t = grp_t["mean"].values

        l1, = ax_l.plot(
            x_t, y_t, "-o",
            color=COLOR_TIME,
            zorder=3,
            label="Sim. time (s)",
        )

        ax_l.set_yscale("log")
        ax_l.set_xscale("log")

        if col_idx == 0:
            ax_l.set_ylabel("Simulation time (s)", color="black")

        grp_m = _agg(df, axis, "peak_mem_mb")
        x_m = grp_m[col].values
        y_m = grp_m["mean"].values / 1024

        l2, = ax_r.plot(
            x_m, y_m, "-^",
            color=COLOR_MEM,
            label="Peak RAM (GiB)",
        )

        ax_r.set_yscale("log")
        ax_r.set_xscale("log")

        _set_plain_log_ticks(ax_r, y_m, n_ticks=5)

        # Show right y-ticks on every subplot
        ax_r.tick_params(axis="y", which="both", right=True, labelright=True)

        # Only put the y-axis label on the last subplot
        if col_idx == 2:
            ax_r.set_ylabel("Peak RAM (GiB)", color="black")

        ax_l.spines["top"].set_visible(True)
        ax_r.spines["top"].set_visible(True)
        ax_r.spines["right"].set_visible(True)

        ax_l.set_xlabel(xlabel, labelpad=3)
        ax_l.set_title(f"Vary {axis}")
        ax_l.grid(axis="y", color="0.88", linewidth=0.5, zorder=0)

        if col_idx == 1:
            ax_l.legend(
                handles=[l1, l2],
                loc="upper left",
                fontsize=7.5,
                framealpha=0.85,
            )

    _save(fig, out_dir / "path_scalability_runtime_memory")


# ── entry point ───────────────────────────────────────────────────────────────

def main(results_path: Path, metric: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {results_path} ...")
    df = load_df(results_path, metric)
    print(f"  {len(df)} valid entries.\n")

    print("Generating figures ...")
    plot_main(df, metric, out_dir)
    plot_runtime_memory(df, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PATH scalability results")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--metric", type=str, default="DS_acc_GRU_TCN",
        choices=["DS_acc_GRU_TCN", "DS_auc_GRU_TCN", "DS_acc_LR_RF", "DS_auc_LR_RF"],
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.results, args.metric, args.out)
