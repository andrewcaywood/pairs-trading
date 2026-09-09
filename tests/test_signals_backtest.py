"""Test signal state transitions and backtest accounting."""

import unittest

import numpy as np
import pandas as pd

from pairs_trading.backtest import run_backtest
from pairs_trading.config import BacktestConfig
from pairs_trading.signals import position_from_zscore, walk_forward_signals


class SignalTests(unittest.TestCase):
    """Verify the trading state machine."""

    def test_positions_enter_and_exit_at_configured_thresholds(self) -> None:
        """Signals should persist until the directional exit is reached."""

        zscore = pd.Series([0.0, -2.1, -1.0, -0.4, 0.0, 2.1, 1.0, 0.4])
        actual = position_from_zscore(zscore, 2.0, 0.5)
        expected = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0])
        pd.testing.assert_series_equal(actual, expected.rename("signal"))

    def test_walk_forward_calibration_uses_only_prior_rows(self) -> None:
        """The first model window must end before the first trade date."""

        generator = np.random.default_rng(7)
        index = pd.bdate_range("2020-01-01", periods=240)
        common = 100 + np.cumsum(generator.normal(0, 0.5, len(index)))
        prices = pd.DataFrame(
            {
                "A": 5 + 1.2 * common + generator.normal(0, 0.1, len(index)),
                "B": common,
            },
            index=index,
        )
        result = walk_forward_signals(
            prices,
            "A",
            "B",
            index[160],
            lookback=120,
            recalibration_frequency=20,
            zscore_window=30,
            entry_z=2.0,
            exit_z=0.5,
            significance_level=0.05,
        )
        first = result.recalibrations.iloc[0]
        self.assertEqual(first["training_end"], index[159])
        self.assertEqual(result.frame.index[0], index[160])
        self.assertTrue(bool(first["cointegrated"]))


class BacktestTests(unittest.TestCase):
    """Verify lagging, costs, and neutrality."""

    def test_backtest_lags_signal_and_charges_costs(self) -> None:
        """A signal should be executed one row later and incur costs."""

        index = pd.bdate_range("2024-01-01", periods=40)
        prices = pd.DataFrame(
            {
                "A": 100 * np.cumprod(np.repeat(1.001, len(index))),
                "B": 100 * np.cumprod(np.repeat(0.999, len(index))),
            },
            index=index,
        )
        signal = pd.Series(0.0, index=index)
        signal.iloc[25:30] = 1.0
        config = BacktestConfig(target_volatility=1.0, max_gross_leverage=1.0)
        result = run_backtest(prices, signal, "A", "B", config)
        self.assertEqual(result.daily["position"].iloc[25], 0.0)
        self.assertEqual(result.daily["position"].iloc[26], 1.0)
        self.assertAlmostEqual(
            abs(result.daily["weight_a"].iloc[26]) + abs(result.daily["weight_b"].iloc[26]),
            1.0,
        )
        self.assertGreater(result.daily["slippage_cost"].sum(), 0.0)
        self.assertGreater(result.daily["commission_cost"].sum(), 0.0)
        self.assertGreater(result.daily["borrow_cost"].sum(), 0.0)
        self.assertLess(result.daily["net_return"].sum(), result.daily["gross_return"].sum())


if __name__ == "__main__":
    unittest.main()
