"""Create the core research figures."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _save(fig: plt.Figure, path: Path) -> Path:
    """Save a figure with consistent output settings."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_cointegration_heatmap(ranking: pd.DataFrame, tickers: list[str], path: Path) -> Path:
    """Plot false-discovery-adjusted cointegration strength by pair."""

    matrix = pd.DataFrame(np.nan, index=tickers, columns=tickers)
    for row in ranking.itertuples():
        strength = -np.log10(max(float(row.coint_qvalue), 1e-12))
        matrix.loc[row.asset_a, row.asset_b] = strength
        matrix.loc[row.asset_b, row.asset_a] = strength
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, cmap="viridis", vmin=0)
    ax.set_xticks(range(len(tickers)), labels=tickers, rotation=45, ha="right")
    ax.set_yticks(range(len(tickers)), labels=tickers)
    ax.set_title("Training-sample cointegration strength")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.set_label("-log10(FDR-adjusted p-value)")
    return _save(fig, path)


def plot_spread_and_zscore(signals: pd.DataFrame, pair_label: str, path: Path) -> Path:
    """Plot the spread and its rolling z-score with trade thresholds."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=[1, 1])
    axes[0].plot(signals.index, signals["spread"], color="#234f72", linewidth=1)
    axes[0].axhline(0, color="black", linewidth=0.7, alpha=0.6)
    axes[0].set_title(f"Walk-forward spread: {pair_label}")
    axes[0].set_ylabel("Regression residual")
    axes[1].plot(signals.index, signals["zscore"], color="#8b3a3a", linewidth=1)
    for threshold in (-2.0, 2.0):
        axes[1].axhline(threshold, color="#b22222", linestyle="--", linewidth=0.8)
    for threshold in (-0.5, 0.0, 0.5):
        axes[1].axhline(threshold, color="gray", linestyle=":" if threshold else "-", linewidth=0.7)
    if "cointegrated" in signals:
        inactive = ~signals["cointegrated"].astype(bool)
        for axis in axes:
            axis.fill_between(
                signals.index,
                0,
                1,
                where=inactive,
                transform=axis.get_xaxis_transform(),
                color="gray",
                alpha=0.12,
                linewidth=0,
            )
    axes[1].set_ylabel("Rolling z-score")
    axes[1].text(
        0.99,
        0.03,
        "Gray periods fail the trailing cointegration test",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        color="dimgray",
        fontsize=9,
    )
    return _save(fig, path)


def plot_equity_and_drawdown(daily: pd.DataFrame, pair_label: str, path: Path) -> Path:
    """Plot comparable equity curves and net-strategy drawdown."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, height_ratios=[2, 1])
    axes[0].plot(daily.index, daily["net_equity"], label="Strategy net", linewidth=1.4)
    axes[0].plot(daily.index, daily["no_cost_equity"], label="Strategy no cost", linewidth=1.1)
    axes[0].plot(daily.index, daily["benchmark_equity"], label="Buy and hold", linewidth=1.1)
    axes[0].set_title(f"Out-of-sample equity: {pair_label}")
    axes[0].set_ylabel("Portfolio value")
    axes[0].legend(frameon=False)
    axes[1].fill_between(daily.index, daily["drawdown"], 0, color="#b22222", alpha=0.7)
    axes[1].set_ylabel("Drawdown")
    return _save(fig, path)
