"""Test market data validation and fixed sample bounds."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pairs_trading.data import load_adjusted_prices


class DataTests(unittest.TestCase):
    """Verify cached price samples are deterministic."""

    def test_cached_data_is_clipped_to_fixed_dates(self) -> None:
        """Rows outside the requested half-open interval should be excluded."""

        index = pd.bdate_range("2019-12-20", "2025-02-10")
        prices = pd.DataFrame({"A": range(1, len(index) + 1)}, index=index)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            prices.to_csv(path)
            result = load_adjusted_prices(
                ["A"],
                "2020-01-01",
                "2025-01-01",
                path,
            )
        self.assertGreaterEqual(result.index.min(), pd.Timestamp("2020-01-01"))
        self.assertLess(result.index.max(), pd.Timestamp("2025-01-01"))


if __name__ == "__main__":
    unittest.main()
