"""Central configuration for the research pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BacktestConfig:
    """Store trading and cost assumptions."""

    zscore_window: int = 60
    volatility_window: int = 20
    entry_z: float = 2.0
    exit_z: float = 0.5
    target_volatility: float = 0.10
    max_gross_leverage: float = 1.0
    commission_per_share: float = 0.005
    slippage_bps: float = 5.0
    annual_borrow_rate: float = 0.02
    annualization_factor: int = 252
    initial_capital: float = 100_000.0


@dataclass(frozen=True)
class ResearchConfig:
    """Store the fixed sample, universe, and validation design."""

    start_date: str = "2010-01-01"
    end_date_inclusive: str = "2026-06-30"
    universe: tuple[str, ...] = (
        "XOM",
        "CVX",
        "COP",
        "EOG",
        "OXY",
        "SLB",
        "HAL",
        "VLO",
        "DVN",
        "APA",
    )
    train_fraction: float = 0.60
    significance_level: float = 0.05
    selection_lookback: int = 756
    walk_forward_lookback: int = 756
    recalibration_frequency: int = 63
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    @property
    def download_end_date(self) -> str:
        """Return the exclusive provider end date."""

        end = __import__("pandas").Timestamp(self.end_date_inclusive)
        return (end + __import__("pandas").Timedelta(days=1)).strftime("%Y-%m-%d")

    @property
    def cache_path(self) -> Path:
        """Return the deterministic adjusted-price cache path."""

        suffix = self.end_date_inclusive.replace("-", "")
        return self.data_dir / f"adjusted_close_{self.start_date[:4]}_{suffix}.csv"
