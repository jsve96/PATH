#!/usr/bin/env python3
"""
Visualise the PATH grid-search results saved in results/path_grid_kdd_cup.json.

Produces:
  1. Heatmaps  — one subplot per kernel, K (rows) x h (cols), coloured by DS_acc
  2. Line plot — DS_acc vs K, one line per kernel, best h per K
  3. Line plot — DS_acc vs h, one line per kernel, best K per h
  4. Bar chart — best DS_acc per kernel
  5. Summary   — best configuration table printed to stdout

Usage:
  python visualize_path_grid.py
  python visualize_path_grid.py --metric DS_auc_GRU_TCN
  python visualize_path_grid.py --results path/to/other.json --out results/img
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_RESULTS = Path("results/path_grid_kdd_cup.json")
DEFAULT_OUT     = Path("results/img")
KERNEL_NAMES    = {0: "quartic", 1: "gaussian", 2: "epanechnikov", 3: "triangular"}
#K_VALUES        = [1, 10, 25, 30, 50, 75, 100]
K_VALUES    = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

H_VALUES        = [0.01,0.025, 0.05, 0.25, 0.1, 0.5]

# ── publication-quality style ─────────────────────────────────────────────────
# Colorblind-safe sequential map anchored to the DS_acc range [0, 0.5].
# "Blues": light (low = good) → dark (high = bad).
# Works in greyscale; no red/green confusion.
HEATMAP_CMAP   = "rocket"
HEATMAP_VMIN   = 0.0
HEATMAP_VMAX   = 0.5

# Colorblind-safe qualitative palette for line / bar plots (Wong 2011)
CB_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
              "#56B4E9", "#D55E00", "#F0E442", "#000000"]


def _setup_style() -> None:
    sns.set_style("ticks")
    mpl.rcParams.update({
        # font
        "font.family":        "sans-serif",
        "font.size":          9,
        "axes.labelsize":     10,
        "axes.titlesize":     11,
        "axes.titleweight":   "bold",
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "legend.framealpha":  0.85,
        "legend.edgecolor":   "0.8",
        # lines
        "lines.linewidth":    1.6,
        "lines.markersize":   5.5,
        # axes
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        # grid
        "grid.linewidth":     0.5,
        "grid.color":         "0.88",
        # output
        "figure.dpi":         120,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
    })


# ── data loading ──────────────────────────────────────────────────────────────

def load_df(results_path: Path, metric: str) -> pd.DataFrame:
    with open(results_path) as f:
        raw = json.load(f)

    records = []
    for entry in raw.values():
        if "error" in entry or "aggregate" not in entry:
            continue
        agg = entry["aggregate"]
        val = agg.get(metric)
        if val is None or val != val:      # skip missing / NaN
            continue
        records.append({
            "K":           entry["K"],
            "h":           entry["h"],
            "kernel_name": entry["kernel_name"],
            "kernel_type": entry["kernel_type"],
            "metric":      float(val),
            "DS_acc":      float(agg.get("DS_acc_GRU_TCN", float("nan"))),
            "DS_auc":      float(agg.get("DS_auc_GRU_TCN", float("nan"))),
        })

    if not records:
        raise ValueError(f"No valid results found in {results_path}.")

    return pd.DataFrame(records)


# ── plot helpers ──────────────────────────────────────────────────────────────

def _heatmap_pivot(df: pd.DataFrame, kernel_name: str) -> pd.DataFrame:
    sub = df[df["kernel_name"] == kernel_name]
    pivot = sub.pivot_table(index="K", columns="h", values="metric", aggfunc="mean")
    return pivot.reindex(index=K_VALUES, columns=sorted(H_VALUES))


def _annot_color(val: float, vmin: float, vmax: float) -> str:
    """Return white for dark cells, near-black for light cells."""
    norm = (val - vmin) / max(vmax - vmin, 1e-9)
    return "white" if norm > 0.60 else "#1a1a1a"


def plot_heatmaps(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    _setup_style()
    kernels = [KERNEL_NAMES[k] for k in sorted(KERNEL_NAMES)]
    n_kern  = len(kernels)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(9.0, 7.6),
        sharey="row",
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    cmap = mpl.colormaps[HEATMAP_CMAP]
    norm = mpl.colors.Normalize(vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX)

    for ax, kernel_name in zip(axes_flat, kernels):
        pivot = _heatmap_pivot(df, kernel_name)

        # draw heatmap without colorbar; we add a shared one below
        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            norm=norm,
            linewidths=0.5,
            linecolor="#d4d4d4",
            annot=False,          # manual annotation for per-cell text colour
            cbar=False,
        )

        # manual per-cell annotation with adaptive text colour
        for row_i, K in enumerate(pivot.index):
            for col_j, h in enumerate(pivot.columns):
                val = pivot.loc[K, h]
                if not np.isnan(val):
                    ax.text(
                        col_j + 0.5, row_i + 0.5,
                        f"{val:.3f}",
                        ha="center", va="center",
                        fontsize=8.5,
                        color=_annot_color(val, HEATMAP_VMIN, HEATMAP_VMAX),
                    )

        # highlight the best (minimum) cell with a thick dark border
        if not pivot.isnull().all().all():
            ri, ci = np.unravel_index(np.nanargmin(pivot.values), pivot.shape)
            rect = mpatches.FancyBboxPatch(
                (ci + 0.04, ri + 0.04), 0.92, 0.92,
                boxstyle="square,pad=0",
                linewidth=2.2,
                edgecolor="#1a1a1a",
                facecolor="none",
                clip_on=False,
                zorder=5,
            )
            ax.add_patch(rect)

        ax.set_title(kernel_name.capitalize())
        ax.set_xlabel("Bandwidth $h$", labelpad=4)
        ax.set_ylabel("History $K$" if ax in axes[:, 0] else "")
        ax.tick_params(length=0)

        # clean up seaborn tick labels
        ax.set_xticklabels([f"{float(l.get_text()):.2g}" for l in ax.get_xticklabels()], rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # shared colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02, aspect=25)
    cbar.set_label(metric, rotation=270, labelpad=14, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # thin horizontal rule separating title region
    fig.suptitle(
        f"PATH hyperparameter grid — KDD Cup  |  metric: {metric}  "
        f"(lower $\\rightarrow$ more realistic;  box = best per kernel)",
        fontsize=9.5,
        y=1.02,
        color="#222222",
    )

    out_path = out_dir / f"path_grid_heatmap_{metric}.pdf"
    fig.savefig(out_path)
    png_path = out_dir / f"path_grid_heatmap_{metric}.png"
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  Saved: {out_path}  +  {png_path}")


def plot_k_effect(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    # one dash style per kernel so lines are distinguishable in b&w
    dashes = [
        (None, None),        # solid
        (6, 1.5),            # long dash
        (2, 1.5),            # short dash
        (6, 1.5, 2, 1.5),   # dash-dot
    ]
    markers = ["o", "s", "^", "D"]

    for (kt, kernel_name), color, dash, marker in zip(
        KERNEL_NAMES.items(), CB_PALETTE, dashes, markers
    ):
        sub = df[df["kernel_name"] == kernel_name]
        if sub.empty:
            continue

        best_h   = sub.groupby("h")["metric"].mean().idxmin()
        line_df  = (sub[sub["h"] == best_h]
                    .groupby("K")["metric"]
                    .mean()
                    .reset_index()
                    .sort_values("K"))

        ls = (0, dash) if dash[0] is not None else "-"
        ax.plot(
            line_df["K"], line_df["metric"],
            marker=marker, linestyle=ls, color=color,
            label=f"{kernel_name}  ($h$={best_h})",
            zorder=3,
        )

    ax.set_xlabel("History $K$")
    ax.set_ylabel(metric)
    ax.set_title(f"Effect of $K$ on {metric}\n(best $h$ per kernel)")
    ax.set_xticks(K_VALUES)
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    sns.despine(ax=ax)

    _save(fig, out_dir / f"path_grid_K_effect_{metric}")


def plot_h_effect(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    dashes  = [(None, None), (6, 1.5), (2, 1.5), (6, 1.5, 2, 1.5)]
    markers = ["o", "s", "^", "D"]

    for (kt, kernel_name), color, dash, marker in zip(
        KERNEL_NAMES.items(), CB_PALETTE, dashes, markers
    ):
        sub = df[df["kernel_name"] == kernel_name]
        if sub.empty:
            continue

        best_K  = sub.groupby("K")["metric"].mean().idxmin()
        line_df = (sub[sub["K"] == best_K]
                   .groupby("h")["metric"]
                   .mean()
                   .reset_index()
                   .sort_values("h"))

        ls = (0, dash) if dash[0] is not None else "-"
        ax.plot(
            line_df["h"], line_df["metric"],
            marker=marker, linestyle=ls, color=color,
            label=f"{kernel_name}  ($K$={best_K})",
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Bandwidth $h$ (log scale)")
    ax.set_ylabel(metric)
    ax.set_title(f"Effect of $h$ on {metric}\n(best $K$ per kernel)")
    ax.set_xticks(sorted(H_VALUES))
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    sns.despine(ax=ax)

    _save(fig, out_dir / f"path_grid_h_effect_{metric}")


def plot_kernel_comparison(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    _setup_style()
    rows = []
    for kt, kernel_name in KERNEL_NAMES.items():
        sub = df[df["kernel_name"] == kernel_name]
        if sub.empty:
            continue
        best = sub.loc[sub["metric"].idxmin()]
        rows.append({
            "kernel":  kernel_name.capitalize(),
            "value":   float(best["metric"]),
            "label":   f"$K$={int(best['K'])},  $h$={best['h']}",
        })

    if not rows:
        return

    summary = pd.DataFrame(rows).sort_values("value")

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    # sequential blue shading: darkest = best (lowest value)
    n   = len(summary)
    pal = [mpl.colormaps["Blues"](0.3 + 0.5 * i / max(n - 1, 1)) for i in range(n)]

    bars = ax.barh(
        summary["kernel"], summary["value"],
        color=pal, edgecolor="white", height=0.55,
        zorder=3,
    )

    for bar, row in zip(bars, summary.itertuples()):
        ax.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            row.label,
            va="center", ha="left",
            fontsize=8, color="#333333",
        )

    max_val = summary["value"].max()
    ax.set_xlim(0, max_val * 1.55)
    ax.set_xlabel(metric)
    ax.set_title(f"Best {metric} per kernel\n" r"($\downarrow$ lower is better)")
    ax.grid(axis="x")
    ax.invert_yaxis()
    sns.despine(ax=ax, left=True)
    ax.tick_params(left=False)

    _save(fig, out_dir / f"path_grid_kernel_comparison_{metric}")


# ── KDE grid ─────────────────────────────────────────────────────────────────

def plot_kde_grid(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    """2x2 grid: one panel per kernel, 2-D KDE in log(h) x K space.

    KDE is weighted by configuration quality (lower metric → higher weight),
    so density peaks at the best region of the hyperparameter space.
    Actual grid points are overlaid as scatter coloured by metric value.
    """
    _setup_style()
    kernels = [KERNEL_NAMES[k] for k in sorted(KERNEL_NAMES)]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.6), constrained_layout=True)
    axes_flat = axes.flatten()

    global_min = df["metric"].min()
    global_max = df["metric"].max()
    log_h_ticks  = np.log10(sorted(H_VALUES))
    h_tick_labels = [f"{h:.2g}" for h in sorted(H_VALUES)]

    cmap_kde  = mpl.colormaps["Blues"]
    cmap_sc   = mpl.colormaps[HEATMAP_CMAP]
    sc_norm   = mpl.colors.Normalize(vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX)

    for ax, kernel_name in zip(axes_flat, kernels):
        sub = df[
            (df["kernel_name"] == kernel_name) &
            (df["K"].isin(K_VALUES)) &
            (df["h"].isin(H_VALUES))
        ].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        log_h = np.log10(sub["h"].values)
        K_vals = sub["K"].values

        # weight: higher = better (lower metric)
        span = max(global_max - global_min, 1e-9)
        weights = (global_max - sub["metric"].values) / span

        # 2-D KDE (only meaningful if we have spread; requires seaborn >= 0.11)
        try:
            sns.kdeplot(
                x=log_h, y=K_vals,
                weights=weights,
                ax=ax,
                fill=True,
                cmap=HEATMAP_CMAP,
                levels=10,
                alpha=0.75,
                bw_adjust=0.75,
            )
        except Exception:
            pass  # fall back to scatter-only if not enough unique points

        # scatter: grid points coloured by metric
        # ax.scatter(
        #     log_h, K_vals,
        #     c=sub["metric"].values,
        #     cmap=cmap_sc, norm=sc_norm,
        #     s=75, zorder=5,
        #     edgecolors="#1a1a1a", linewidths=0.55,
        # )

        ax.set_title(kernel_name.capitalize())
        ax.set_xlabel("$h$", labelpad=4)
        ax.set_ylabel("$K$" if ax in axes[:, 0] else "")

        # pin axes to exact data range — no KDE bleed-out padding
        x_margin = (log_h_ticks[-1] - log_h_ticks[0]) * 0.04
        y_margin = (K_VALUES[-1] - K_VALUES[0]) * 0.04
        ax.set_xlim(log_h_ticks[0] - x_margin, log_h_ticks[-1] + x_margin)
        ax.set_ylim(K_VALUES[0] - y_margin, K_VALUES[-1] + y_margin)

        ax.set_xticks(log_h_ticks)
        ax.set_xticklabels(h_tick_labels, rotation=0)
        ax.set_yticks(K_VALUES)
        ax.tick_params(length=0)

        sns.despine(ax=ax)

    # colorbar: rocket_r so light = low DS_acc (good), matching the KDE fill
    sm = mpl.cm.ScalarMappable(
        cmap=mpl.colormaps["rocket_r"],
        norm=mpl.colors.Normalize(vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02, aspect=25)
    cbar.set_label(metric, rotation=270, labelpad=14, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # fig.suptitle(
    #     f"PATH hyperparameter density — KDD Cup  |  metric: {metric}\n"
    #     fontsize=9.5, y=1.02, color="#222222",
    # )

    _save(fig, out_dir / f"path_grid_kde_{metric}")


# ── shared save helper ────────────────────────────────────────────────────────

def _save(fig: plt.Figure, stem: Path) -> None:
    for ext in (".pdf", ".png"):
        p = stem.with_suffix(ext)
        fig.savefig(p)
        print(f"  Saved: {p}")
    plt.close(fig)


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, metric: str) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Summary — {metric}  (lower = better)")
    print(sep)

    best = df.loc[df["metric"].idxmin()]
    print(f"\n  Global best:")
    print(f"    kernel={best['kernel_name']}  K={int(best['K'])}  "
          f"h={best['h']}  {metric}={best['metric']:.4f}")

    print(f"\n  Best per kernel:")
    for kernel_name in [KERNEL_NAMES[k] for k in sorted(KERNEL_NAMES)]:
        sub = df[df["kernel_name"] == kernel_name]
        if sub.empty:
            print(f"    {kernel_name:<16} — no data")
            continue
        row = sub.loc[sub["metric"].idxmin()]
        print(f"    {kernel_name:<16}  K={int(row['K']):<4}  h={row['h']:<5}  "
              f"{metric}={row['metric']:.4f}")

    print(f"\n  Configurations evaluated: {len(df)}")
    print(f"{sep}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main(results_path: Path, metric: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {results_path} ...")
    df = load_df(results_path, metric)
    print(f"  {len(df)} valid configurations.\n")

    print("Generating figures ...")
    plot_heatmaps(df, metric, out_dir)
    plot_kde_grid(df, metric, out_dir)
    plot_k_effect(df, metric, out_dir)
    plot_h_effect(df, metric, out_dir)
    plot_kernel_comparison(df, metric, out_dir)

    print_summary(df, metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PATH grid-search results")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--metric", type=str, default="DS_acc_GRU_TCN",
        choices=["DS_acc_GRU_TCN", "DS_auc_GRU_TCN", "DS_acc_LR_RF", "DS_auc_LR_RF"],
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.results, args.metric, args.out)
