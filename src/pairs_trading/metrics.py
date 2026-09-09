"""Calculate strategy and subperiod performance statistics."""

import numpy as np
import pandas as pd


def performance_metrics(
    returns: pd.Series,
    trades: pd.DataFrame | None = None,
    annualization_factor: int = 252,
) -> dict[str, float | int]:
    """Calculate standard return, risk, drawdown, and trade metrics."""

    clean = returns.dropna().astype(float)
    if clean.empty:
        raise ValueError("Returns must contain at least one observation.")
    equity = (1 + clean).cumprod()
    years = len(clean) / annualization_factor
    total_return = float(equity.iloc[-1] - 1)
    annualized_return = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else float("nan")
    annualized_volatility = float(clean.std(ddof=1) * np.sqrt(annualization_factor))
    sharpe = (
        float(clean.mean() / clean.std(ddof=1) * np.sqrt(annualization_factor))
        if clean.std(ddof=1) > 0
        else float("nan")
    )
    drawdown = equity / equity.cummax() - 1
    closed = trades.dropna(subset=["exit_date"]) if trades is not None and not trades.empty else None
    return {
        "observations": len(clean),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "maximum_drawdown": float(drawdown.min()),
        "trade_count": int(len(closed)) if closed is not None else 0,
        "win_rate": float((closed["net_return"] > 0).mean()) if closed is not None and len(closed) else float("nan"),
    }


def comparison_metrics(daily: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Compare net, no-cost, and buy-and-hold performance."""

    mappings = {
        "strategy_net": "net_return",
        "strategy_no_cost": "gross_return",
        "buy_and_hold": "benchmark_return",
    }
    rows = []
    for label, column in mappings.items():
        result = performance_metrics(daily[column], trades if label == "strategy_net" else None)
        rows.append({"series": label, **result})
    return pd.DataFrame(rows).set_index("series")


def subperiod_metrics(
    daily: pd.DataFrame,
    periods: dict[str, tuple[str, str]],
    return_column: str = "net_return",
    trades: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate performance over user-defined date regimes."""

    rows = []
    for label, (start, end) in periods.items():
        sample = daily.loc[start:end, return_column]
        if not sample.empty:
            period_trades = None
            if trades is not None and not trades.empty:
                entries = pd.to_datetime(trades["entry_date"])
                period_trades = trades[(entries >= pd.Timestamp(start)) & (entries <= pd.Timestamp(end))]
            rows.append({"period": label, **performance_metrics(sample, period_trades)})
    return pd.DataFrame(rows).set_index("period") if rows else pd.DataFrame()


def cost_summary(daily: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """Summarize modeled trading frictions over the test period."""

    components = {
        "slippage": float(daily["slippage_cost"].sum()),
        "commission": float(daily["commission_cost"].sum()),
        "short_borrow": float(daily["borrow_cost"].sum()),
    }
    rows = [
        {
            "component": label,
            "return_drag": value,
            "initial_capital_equivalent": value * initial_capital,
        }
        for label, value in components.items()
    ]
    total = sum(components.values())
    rows.append(
        {
            "component": "total",
            "return_drag": total,
            "initial_capital_equivalent": total * initial_capital,
        }
    )
    return pd.DataFrame(rows).set_index("component")
