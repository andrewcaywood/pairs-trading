"""Run a point-in-time, cost-aware pairs trading backtest."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pairs_trading.config import BacktestConfig


@dataclass(frozen=True)
class BacktestResult:
    """Bundle daily results and the closed-trade ledger."""

    daily: pd.DataFrame
    trades: pd.DataFrame


def _trade_ledger(daily: pd.DataFrame) -> pd.DataFrame:
    """Summarize contiguous nonzero positions as individual trades."""

    rows: list[dict[str, object]] = []
    active: dict[str, object] | None = None
    for date, row in daily.iterrows():
        position = float(row["position"])
        previous = float(row["position_previous"])
        if active is not None:
            active["daily_returns"].append(float(row["net_return"]))
        if previous != 0 and position != previous:
            if active is not None:
                values = active.pop("daily_returns")
                active["exit_date"] = date
                active["holding_days"] = len(values)
                active["net_return"] = float(np.prod(np.asarray(values) + 1) - 1)
                rows.append(active)
                active = None
        if position != 0 and position != previous:
            active = {
                "entry_date": date,
                "side": "long_spread" if position > 0 else "short_spread",
                "daily_returns": [float(row["net_return"])],
            }
    if active is not None:
        values = active.pop("daily_returns")
        active["exit_date"] = pd.NaT
        active["holding_days"] = len(values)
        active["net_return"] = float(np.prod(np.asarray(values) + 1) - 1)
        rows.append(active)
    columns = ["entry_date", "exit_date", "side", "holding_days", "net_return"]
    return pd.DataFrame(rows, columns=columns)


def run_backtest(
    prices: pd.DataFrame,
    signal: pd.Series,
    asset_a: str,
    asset_b: str,
    config: BacktestConfig,
) -> BacktestResult:
    """Backtest equal-dollar legs with volatility scaling and explicit costs."""

    pair = prices[[asset_a, asset_b]].dropna()
    returns = pair.pct_change(fill_method=None).fillna(0.0)
    aligned_signal = signal.reindex(pair.index).fillna(0.0)

    spread_return = returns[asset_a] - returns[asset_b]
    realized_volatility = (
        spread_return.rolling(config.volatility_window, min_periods=config.volatility_window)
        .std(ddof=1)
        .shift(1)
        * np.sqrt(config.annualization_factor)
    )
    gross_scale = (config.target_volatility / realized_volatility).clip(
        lower=0.0,
        upper=config.max_gross_leverage,
    )
    gross_scale = gross_scale.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    position = aligned_signal.shift(1).fillna(0.0)
    weight_a = position * gross_scale * 0.5
    weight_b = -position * gross_scale * 0.5
    delta_a = weight_a.diff().fillna(weight_a)
    delta_b = weight_b.diff().fillna(weight_b)
    turnover = delta_a.abs() + delta_b.abs()

    gross_return = weight_a * returns[asset_a] + weight_b * returns[asset_b]
    slippage_cost = turnover * config.slippage_bps / 10_000
    commission_cost = (
        delta_a.abs() * config.commission_per_share / pair[asset_a]
        + delta_b.abs() * config.commission_per_share / pair[asset_b]
    )
    short_exposure = (-weight_a.clip(upper=0)) + (-weight_b.clip(upper=0))
    borrow_cost = short_exposure * config.annual_borrow_rate / config.annualization_factor
    net_return = gross_return - slippage_cost - commission_cost - borrow_cost

    daily = pd.DataFrame(
        {
            "price_a": pair[asset_a],
            "price_b": pair[asset_b],
            "signal": aligned_signal,
            "position": position,
            "position_previous": position.shift(1).fillna(0.0),
            "realized_spread_volatility": realized_volatility,
            "gross_scale": gross_scale,
            "weight_a": weight_a,
            "weight_b": weight_b,
            "turnover": turnover,
            "gross_return": gross_return,
            "slippage_cost": slippage_cost,
            "commission_cost": commission_cost,
            "borrow_cost": borrow_cost,
            "net_return": net_return,
        }
    )
    daily["no_cost_equity"] = config.initial_capital * (1 + daily["gross_return"]).cumprod()
    daily["net_equity"] = config.initial_capital * (1 + daily["net_return"]).cumprod()
    benchmark_return = 0.5 * returns[asset_a] + 0.5 * returns[asset_b]
    daily["benchmark_return"] = benchmark_return
    daily["benchmark_equity"] = config.initial_capital * (1 + benchmark_return).cumprod()
    daily["drawdown"] = daily["net_equity"] / daily["net_equity"].cummax() - 1
    return BacktestResult(daily=daily, trades=_trade_ledger(daily))

