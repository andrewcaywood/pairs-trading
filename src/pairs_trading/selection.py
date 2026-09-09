"""Estimate hedge relationships and rank candidate pairs."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


@dataclass(frozen=True)
class HedgeModel:
    """Represent a linear price relationship."""

    asset_a: str
    asset_b: str
    intercept: float
    hedge_ratio: float

    def spread(self, prices: pd.DataFrame) -> pd.Series:
        """Calculate the regression residual spread."""

        return prices[self.asset_a] - self.intercept - self.hedge_ratio * prices[self.asset_b]


def fit_hedge_model(prices: pd.DataFrame, asset_a: str, asset_b: str) -> HedgeModel:
    """Fit an OLS hedge relationship using aligned price levels."""

    sample = prices[[asset_a, asset_b]].dropna()
    if len(sample) < 30:
        raise ValueError("At least 30 aligned observations are required.")
    design = sm.add_constant(sample[asset_b], has_constant="add")
    fit = sm.OLS(sample[asset_a], design).fit()
    return HedgeModel(asset_a, asset_b, float(fit.params["const"]), float(fit.params[asset_b]))


def _half_life(spread: pd.Series) -> float:
    """Estimate spread mean-reversion half-life in trading days."""

    lagged = spread.shift(1)
    delta = spread.diff()
    sample = pd.concat([delta.rename("delta"), lagged.rename("lagged")], axis=1).dropna()
    if len(sample) < 30:
        return float("nan")
    fit = sm.OLS(sample["delta"], sm.add_constant(sample["lagged"])).fit()
    speed = float(fit.params["lagged"])
    return float(-np.log(2) / speed) if speed < 0 else float("inf")


def _fdr_adjust(pvalues: pd.Series) -> pd.Series:
    """Apply the Benjamini-Hochberg false-discovery correction."""

    values = pvalues.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return pd.Series(output, index=pvalues.index)


def evaluate_pair(prices: pd.DataFrame, asset_a: str, asset_b: str) -> dict[str, float | str | int]:
    """Calculate cointegration, stationarity, and relationship diagnostics."""

    sample = prices[[asset_a, asset_b]].dropna()
    model = fit_hedge_model(sample, asset_a, asset_b)
    spread = model.spread(sample)
    coint_stat, coint_pvalue, _ = coint(
        sample[asset_a], sample[asset_b], trend="c", autolag="aic"
    )
    adf_result = adfuller(spread, regression="c", autolag="AIC", result_object=True)
    adf_stat, adf_pvalue = adf_result[0], adf_result[1]
    returns = sample.pct_change(fill_method=None).dropna()
    return {
        "asset_a": asset_a,
        "asset_b": asset_b,
        "observations": len(sample),
        "price_correlation": float(sample[asset_a].corr(sample[asset_b])),
        "return_correlation": float(returns[asset_a].corr(returns[asset_b])),
        "intercept": model.intercept,
        "hedge_ratio": model.hedge_ratio,
        "coint_stat": float(coint_stat),
        "coint_pvalue": float(coint_pvalue),
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "half_life_days": _half_life(spread),
    }


def rank_pairs(prices: pd.DataFrame) -> pd.DataFrame:
    """Evaluate every pair and rank results by cointegration strength."""

    rows = [evaluate_pair(prices, a, b) for a, b in combinations(prices.columns, 2)]
    ranking = pd.DataFrame(rows)
    ranking["coint_qvalue"] = _fdr_adjust(ranking["coint_pvalue"])
    ranking["adf_qvalue"] = _fdr_adjust(ranking["adf_pvalue"])
    return ranking.sort_values(
        ["coint_qvalue", "coint_pvalue", "half_life_days"],
        kind="stable",
    ).reset_index(drop=True)


def choose_pair(ranking: pd.DataFrame, significance_level: float = 0.05) -> pd.Series:
    """Choose the strongest pair that passes both residual tests."""

    eligible = ranking[
        (ranking["coint_pvalue"] <= significance_level)
        & (ranking["adf_pvalue"] <= significance_level)
        & (ranking["hedge_ratio"] > 0)
        & np.isfinite(ranking["half_life_days"])
        & (ranking["half_life_days"] > 0)
    ]
    if eligible.empty:
        raise ValueError("No pair passes the configured cointegration criteria.")
    return eligible.iloc[0]
