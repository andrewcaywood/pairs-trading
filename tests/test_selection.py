"""Test pair estimation and ranking utilities."""

import unittest

import numpy as np
import pandas as pd

from pairs_trading.selection import choose_pair, fit_hedge_model, rank_pairs


class SelectionTests(unittest.TestCase):
    """Verify hedge estimation and cointegration selection."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create deterministic cointegrated and random-walk samples."""

        generator = np.random.default_rng(42)
        index = pd.bdate_range("2010-01-01", periods=1200)
        common = 100 + np.cumsum(generator.normal(0, 1, len(index)))
        cls.prices = pd.DataFrame(
            {
                "A": 10 + 1.5 * common + generator.normal(0, 0.4, len(index)),
                "B": common,
                "C": 80 + np.cumsum(generator.normal(0, 1, len(index))),
            },
            index=index,
        )

    def test_hedge_ratio_is_recovered(self) -> None:
        """OLS should recover the simulated hedge ratio."""

        model = fit_hedge_model(self.prices, "A", "B")
        self.assertAlmostEqual(model.hedge_ratio, 1.5, places=2)

    def test_cointegrated_pair_ranks_first(self) -> None:
        """The known cointegrated pair should survive correction."""

        ranking = rank_pairs(self.prices)
        winner = choose_pair(ranking)
        self.assertEqual((winner["asset_a"], winner["asset_b"]), ("A", "B"))


if __name__ == "__main__":
    unittest.main()

