"""Run pair selection and out-of-sample backtests."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pairs_trading.backtest import run_backtest
from pairs_trading.config import BacktestConfig, ResearchConfig
from pairs_trading.data import load_adjusted_prices
from pairs_trading.metrics import comparison_metrics, cost_summary, subperiod_metrics
from pairs_trading.plots import (
    plot_cointegration_heatmap,
    plot_equity_and_drawdown,
    plot_spread_and_zscore,
)
from pairs_trading.selection import choose_pair, fit_hedge_model, rank_pairs
from pairs_trading.signals import static_signals, walk_forward_signals


def _write_json(frame: pd.DataFrame, path: Path) -> None:
    """Write a metric frame as readable JSON."""

    payload = json.loads(frame.reset_index().to_json(orient="records", date_format="iso"))
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="ascii")


def run(refresh: bool = False) -> dict[str, object]:
    """Execute the full fixed-sample research pipeline."""

    research = ResearchConfig(data_dir=PROJECT_ROOT / "data", output_dir=PROJECT_ROOT / "outputs")
    backtest_config = BacktestConfig()
    prices = load_adjusted_prices(
        research.universe,
        research.start_date,
        research.download_end_date,
        research.cache_path,
        refresh=refresh,
    )
    split_location = int(len(prices) * research.train_fraction)
    training = prices.iloc[:split_location]
    test_start = prices.index[split_location]
    selection_sample = training.iloc[-research.selection_lookback :]

    ranking = rank_pairs(selection_sample)
    winner = choose_pair(ranking, research.significance_level)
    asset_a = str(winner["asset_a"])
    asset_b = str(winner["asset_b"])
    model = fit_hedge_model(selection_sample, asset_a, asset_b)

    static_signal = static_signals(
        prices,
        model,
        backtest_config.zscore_window,
        backtest_config.entry_z,
        backtest_config.exit_z,
    ).frame.loc[test_start:]
    static_result = run_backtest(
        prices.loc[test_start:], static_signal["signal"], asset_a, asset_b, backtest_config
    )

    walk_forward = walk_forward_signals(
        prices,
        asset_a,
        asset_b,
        test_start,
        research.walk_forward_lookback,
        research.recalibration_frequency,
        backtest_config.zscore_window,
        backtest_config.entry_z,
        backtest_config.exit_z,
        research.significance_level,
    )
    walk_result = run_backtest(
        prices.loc[test_start:], walk_forward.frame["signal"], asset_a, asset_b, backtest_config
    )

    output = research.output_dir
    figure_dir = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output / "pair_ranking.csv", index=False, float_format="%.8f")
    static_result.daily.to_csv(output / "static_daily.csv", index_label="date", float_format="%.10f")
    static_result.trades.to_csv(output / "static_trades.csv", index=False, float_format="%.8f")
    walk_result.daily.join(walk_forward.frame.add_prefix("model_")).to_csv(
        output / "walk_forward_daily.csv", index_label="date", float_format="%.10f"
    )
    walk_result.trades.to_csv(output / "walk_forward_trades.csv", index=False, float_format="%.8f")
    walk_forward.recalibrations.to_csv(
        output / "walk_forward_recalibrations.csv", index_label="date", float_format="%.8f"
    )

    static_metrics = comparison_metrics(static_result.daily, static_result.trades)
    walk_metrics = comparison_metrics(walk_result.daily, walk_result.trades)
    periods = {
        "covid_shock_2020_2021": ("2020-01-01", "2021-12-31"),
        "energy_shock_2022_2023": ("2022-01-01", "2023-12-31"),
        "recent_2024_2026": ("2024-01-01", research.end_date_inclusive),
    }
    regime_metrics = subperiod_metrics(walk_result.daily, periods, trades=walk_result.trades)
    costs = cost_summary(walk_result.daily, backtest_config.initial_capital)
    _write_json(static_metrics, output / "static_metrics.json")
    _write_json(walk_metrics, output / "walk_forward_metrics.json")
    _write_json(regime_metrics, output / "regime_metrics.json")
    _write_json(costs, output / "cost_summary.json")

    price_hash = hashlib.sha256(research.cache_path.read_bytes()).hexdigest()
    metadata = {
        "data_source": "Yahoo Finance adjusted daily close",
        "data_start": str(prices.index[0].date()),
        "data_end": str(prices.index[-1].date()),
        "data_sha256": price_hash,
        "research_config": asdict(research),
        "backtest_config": asdict(backtest_config),
        "selected_pair": [asset_a, asset_b],
        "selection_sample_start": str(selection_sample.index[0].date()),
        "selection_sample_end": str(selection_sample.index[-1].date()),
        "test_start": str(test_start.date()),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="ascii",
    )

    pair_label = f"{asset_a}/{asset_b}"
    plot_cointegration_heatmap(ranking, list(research.universe), figure_dir / "cointegration_heatmap.png")
    plot_spread_and_zscore(walk_forward.frame, pair_label, figure_dir / "spread_zscore.png")
    plot_equity_and_drawdown(walk_result.daily, pair_label, figure_dir / "equity_drawdown.png")

    summary = {
        "pair": pair_label,
        "training_start": str(training.index[0].date()),
        "training_end": str(training.index[-1].date()),
        "selection_start": str(selection_sample.index[0].date()),
        "test_start": str(test_start.date()),
        "test_end": str(prices.index[-1].date()),
        "ranking_row": winner.to_dict(),
        "static_metrics": static_metrics.reset_index().to_dict(orient="records"),
        "walk_forward_metrics": walk_metrics.reset_index().to_dict(orient="records"),
    }
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    arguments = parser.parse_args()
    run(refresh=arguments.refresh)
