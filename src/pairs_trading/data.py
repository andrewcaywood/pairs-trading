"""Download, cache, and validate adjusted daily prices."""

from pathlib import Path
from typing import Iterable

import pandas as pd


def _extract_close(download: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalize yfinance output into one adjusted-close column per ticker."""

    if download.empty:
        raise ValueError("The market data download returned no rows.")
    if isinstance(download.columns, pd.MultiIndex):
        level_zero = download.columns.get_level_values(0)
        if "Close" not in level_zero:
            raise ValueError("The download does not contain adjusted close prices.")
        close = download["Close"].copy()
    else:
        if len(tickers) != 1 or "Close" not in download.columns:
            raise ValueError("The download has an unexpected column layout.")
        close = download[["Close"]].rename(columns={"Close": tickers[0]})
    return close.reindex(columns=tickers).sort_index()


def validate_prices(
    prices: pd.DataFrame,
    tickers: Iterable[str],
    minimum_observations: int = 500,
) -> pd.DataFrame:
    """Validate price coverage and return a clean float frame."""

    expected = list(tickers)
    missing = sorted(set(expected) - set(prices.columns))
    if missing:
        raise ValueError(f"Missing price columns: {', '.join(missing)}")
    clean = prices.loc[:, expected].copy()
    clean.index = pd.to_datetime(clean.index, utc=True).tz_localize(None)
    clean = clean[~clean.index.duplicated(keep="last")].sort_index().astype(float)
    clean = clean.where(clean > 0)
    insufficient = clean.count()[clean.count() < minimum_observations]
    if not insufficient.empty:
        labels = ", ".join(f"{key}={value}" for key, value in insufficient.items())
        raise ValueError(f"Insufficient observations: {labels}")
    return clean


def _bound_sample(prices: pd.DataFrame, start: str, end_exclusive: str) -> pd.DataFrame:
    """Restrict prices to the configured half-open date interval."""

    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end_exclusive)
    bounded = prices[(prices.index >= start_date) & (prices.index < end_date)]
    if bounded.empty:
        raise ValueError("No prices fall inside the configured sample period.")
    return bounded


def load_adjusted_prices(
    tickers: Iterable[str],
    start: str,
    end_exclusive: str,
    cache_path: str | Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load cached prices or download a fixed adjusted-close sample."""

    ticker_list = list(tickers)
    path = Path(cache_path)
    if path.exists() and not refresh:
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        if set(ticker_list).issubset(cached.columns):
            clean = validate_prices(cached, ticker_list)
            return _bound_sample(clean, start, end_exclusive)

    import yfinance as yf

    provider_cache = path.parent / ".yfinance_cache"
    provider_cache.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(provider_cache))
    downloaded = yf.download(
        ticker_list,
        start=start,
        end=end_exclusive,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    prices = validate_prices(_extract_close(downloaded, ticker_list), ticker_list)
    prices = _bound_sample(prices, start, end_exclusive)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index_label="date", float_format="%.8f")
    return prices
