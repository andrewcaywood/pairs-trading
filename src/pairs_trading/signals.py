"""Construct static and walk-forward spreads and trading signals."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pairs_trading.selection import HedgeModel, evaluate_pair


@dataclass(frozen=True)
class SignalFrame:
    """Bundle point-in-time model values and signals."""

    frame: pd.DataFrame
    recalibrations: pd.DataFrame


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Calculate a rolling z-score including only current and prior values."""

    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=1)
    return (series - mean) / std.replace(0, np.nan)


def position_from_zscore(zscore: pd.Series, entry_z: float, exit_z: float) -> pd.Series:
    """Convert z-scores into a persistent mean-reversion position state."""

    if not 0 <= exit_z < entry_z:
        raise ValueError("Thresholds must satisfy 0 <= exit_z < entry_z.")
    states: list[float] = []
    state = 0.0
    for value in zscore:
        if not np.isfinite(value):
            state = 0.0
        elif state == 0.0:
            if value <= -entry_z:
                state = 1.0
            elif value >= entry_z:
                state = -1.0
        elif state > 0 and value >= -exit_z:
            state = 0.0
        elif state < 0 and value <= exit_z:
            state = 0.0
        states.append(state)
    return pd.Series(states, index=zscore.index, name="signal")


def static_signals(
    prices: pd.DataFrame,
    model: HedgeModel,
    zscore_window: int,
    entry_z: float,
    exit_z: float,
) -> SignalFrame:
    """Create signals from one hedge model estimated before the test period."""

    spread = model.spread(prices)
    zscore = rolling_zscore(spread, zscore_window)
    signal = position_from_zscore(zscore, entry_z, exit_z)
    frame = pd.DataFrame(
        {
            "intercept": model.intercept,
            "hedge_ratio": model.hedge_ratio,
            "spread": spread,
            "zscore": zscore,
            "signal": signal,
        }
    )
    return SignalFrame(frame=frame, recalibrations=pd.DataFrame())


def walk_forward_signals(
    prices: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    first_trade_date: pd.Timestamp,
    lookback: int,
    recalibration_frequency: int,
    zscore_window: int,
    entry_z: float,
    exit_z: float,
    significance_level: float,
) -> SignalFrame:
    """Re-estimate the hedge model periodically using trailing data only."""

    pair = prices[[asset_a, asset_b]].dropna()
    first_location = int(pair.index.searchsorted(first_trade_date))
    if first_location < lookback:
        raise ValueError("The first trade date does not have enough lookback history.")

    records: list[pd.DataFrame] = []
    calibrations: list[dict[str, float | pd.Timestamp]] = []
    for start_location in range(first_location, len(pair), recalibration_frequency):
        stop_location = min(start_location + recalibration_frequency, len(pair))
        training = pair.iloc[start_location - lookback : start_location]
        diagnostics = evaluate_pair(training, asset_a, asset_b)
        model = HedgeModel(
            asset_a,
            asset_b,
            float(diagnostics["intercept"]),
            float(diagnostics["hedge_ratio"]),
        )
        cointegrated = bool(
            diagnostics["coint_pvalue"] <= significance_level
            and diagnostics["adf_pvalue"] <= significance_level
            and diagnostics["hedge_ratio"] > 0
        )
        segment_start = max(0, start_location - zscore_window + 1)
        context = pair.iloc[segment_start:stop_location]
        spread = model.spread(context)
        zscore = rolling_zscore(spread, zscore_window).iloc[-(stop_location - start_location) :]
        segment = pd.DataFrame(index=pair.index[start_location:stop_location])
        segment["intercept"] = model.intercept
        segment["hedge_ratio"] = model.hedge_ratio
        segment["spread"] = spread.reindex(segment.index)
        segment["zscore"] = zscore.reindex(segment.index)
        segment["cointegrated"] = cointegrated
        records.append(segment)
        calibrations.append(
            {
                "date": pair.index[start_location],
                "training_start": training.index[0],
                "training_end": training.index[-1],
                "intercept": model.intercept,
                "hedge_ratio": model.hedge_ratio,
                "coint_stat": diagnostics["coint_stat"],
                "coint_pvalue": diagnostics["coint_pvalue"],
                "adf_stat": diagnostics["adf_stat"],
                "adf_pvalue": diagnostics["adf_pvalue"],
                "half_life_days": diagnostics["half_life_days"],
                "cointegrated": cointegrated,
            }
        )
    frame = pd.concat(records).sort_index()
    eligible_zscore = frame["zscore"].where(frame["cointegrated"])
    frame["signal"] = position_from_zscore(eligible_zscore, entry_z, exit_z)
    return SignalFrame(frame=frame, recalibrations=pd.DataFrame(calibrations).set_index("date"))
