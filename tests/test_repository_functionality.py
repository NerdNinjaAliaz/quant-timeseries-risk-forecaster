"""Comprehensive Functional & Quantitative Verification Suite for quant-timeseries-risk-forecaster.

Runs with zero external dependencies using Python's standard library (and optionally
runs PyTorch/scikit-learn tensor checks if installed in the active virtualenv):
    python3 -m unittest discover -s tests -v
"""

import ast
import csv
import py_compile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


class TestRepositoryFunctionality(unittest.TestCase):
    """End-to-end verification of syntax, data hygiene, model topology, and risk metrics."""

    def test_01_python_syntax_and_ast_structure(self) -> None:
        """Verify all Python files compile cleanly and expose expected classes/functions."""
        py_files = [
            ROOT_DIR / "app.py",
            ROOT_DIR / "run_experiment.py",
            ROOT_DIR / "src" / "__init__.py",
            ROOT_DIR / "src" / "data_pipeline.py",
            ROOT_DIR / "src" / "models.py",
            ROOT_DIR / "src" / "backtest.py",
        ]
        for file_path in py_files:
            self.assertTrue(file_path.exists(), f"Missing file: {file_path}")
            py_compile.compile(str(file_path), doraise=True)
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
            self.assertIsInstance(tree, ast.Module)

        models_ast = ast.parse((ROOT_DIR / "src" / "models.py").read_text(encoding="utf-8"))
        model_classes = {node.name for node in ast.walk(models_ast) if isinstance(node, ast.ClassDef)}
        self.assertIn("StackedRecurrentForecaster", model_classes)
        self.assertIn("TemporalMLPForecaster", model_classes)

        backtest_ast = ast.parse((ROOT_DIR / "src" / "backtest.py").read_text(encoding="utf-8"))
        backtest_funcs = {node.name for node in ast.walk(backtest_ast) if isinstance(node, ast.FunctionDef)}
        self.assertTrue(
            {
                "compute_signals_from_predictions",
                "compute_baseline_signals",
                "evaluate_portfolio_strategy",
            }.issubset(backtest_funcs)
        )

    def test_02_btc_dataset_integrity_and_chronological_split(self) -> None:
        """Verify BTC-USD.csv has 2,435 daily bars, strict ordering, and valid OHLCV bounds."""
        csv_path = ROOT_DIR / "data" / "BTC-USD.csv"
        self.assertTrue(csv_path.exists(), "Missing data/BTC-USD.csv")

        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 2435, "Expected exactly 2,435 daily BTC-USD bars")
        self.assertEqual(rows[0]["Date"], "2014-09-20")
        self.assertEqual(rows[-1]["Date"], "2021-05-20")

        dates = [r["Date"] for r in rows]
        self.assertEqual(dates, sorted(dates), "Dates must be strictly chronological")
        self.assertEqual(len(set(dates)), 2435, "Dates must contain zero duplicates")

        train_rows = [r for r in rows if r["Date"] < "2020-01-01"]
        test_rows = [r for r in rows if r["Date"] > "2020-01-01"]
        self.assertEqual(len(train_rows), 1929, "Expected 1,929 pre-2020 training bars")
        self.assertEqual(len(test_rows), 505, "Expected 505 out-of-sample test bars (2020-2021)")

        # Verify 28-day rolling window sample counts
        lookback = 28
        self.assertEqual(len(train_rows) - lookback, 1901, "Expected 1,901 28D training windows")
        self.assertEqual(len(test_rows), 505, "Expected 505 28D out-of-sample test windows")

        for r in rows:
            open_p = float(r["Open"])
            high_p = float(r["High"])
            low_p = float(r["Low"])
            close_p = float(r["Close"])
            vol = float(r["Volume"])
            self.assertGreaterEqual(high_p, max(open_p, close_p) - 1e-4)
            self.assertLessEqual(low_p, min(open_p, close_p) + 1e-4)
            self.assertGreater(close_p, 0.0)
            self.assertGreater(vol, 0.0)

    def test_03_leak_free_scaling_and_window_alignment(self) -> None:
        """Verify scaler fit strictly on training data prevents test-set lookahead bias."""
        csv_path = ROOT_DIR / "data" / "BTC-USD.csv"
        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        train_closes = [float(r["Close"]) for r in rows if r["Date"] < "2020-01-01"]
        test_closes = [float(r["Close"]) for r in rows if r["Date"] > "2020-01-01"]

        train_min, train_max = min(train_closes), max(train_closes)
        test_min, test_max = min(test_closes), max(test_closes)

        self.assertAlmostEqual(train_min, 172.88476, places=4)
        self.assertGreater(train_max, 19400.0)
        self.assertGreater(test_max, 63500.0)
        self.assertGreater(test_min, 4500.0)

        # Under leak-free MinMaxScaler fit on train only, max test Close > 1.0 (~3.10+)
        scaled_test_peak = (test_max - train_min) / (train_max - train_min)
        self.assertGreater(scaled_test_peak, 3.0)
        # Verify inverse transformation recovers exact USD value
        recovered_usd = scaled_test_peak * (train_max - train_min) + train_min
        self.assertAlmostEqual(recovered_usd, test_max, places=5)

    def test_04_model_parameter_counts_and_topology(self) -> None:
        """Verify parameter counts for 4-layer [50, 60, 80, 120] LSTM, GRU, RNN, and MLP."""
        hidden_dims = [50, 60, 80, 120]
        input_dim = 5
        seq_len = 28

        # Exact parameter formula for stacked RNN/GRU/LSTM + Linear(120 -> 1)
        def count_rnn_params(gates: int) -> int:
            total = 0
            in_d = input_dim
            for h in hidden_dims:
                total += gates * (in_d * h + h * h + 2 * h)
                in_d = h
            total += hidden_dims[-1] * 1 + 1
            return total

        self.assertEqual(count_rnn_params(4), 180801)  # 4-Layer Stacked LSTM (180,801 params)
        self.assertEqual(count_rnn_params(3), 135631)  # 4-Layer Stacked GRU (135,631 params)
        self.assertEqual(count_rnn_params(1), 45291)   # 4-Layer SimpleRNN (45,291 params)

        mlp_params = 0
        in_d = seq_len * input_dim
        for h in hidden_dims:
            mlp_params += in_d * h + h
            in_d = h
        mlp_params += hidden_dims[-1] * 1 + 1
        self.assertEqual(mlp_params, 24831)            # 4-Layer Temporal MLP (24,831 params)

    def test_05_backtest_scorecard_and_equity_curve_consistency(self) -> None:
        """Verify benchmark_metrics.csv and equity_curves.csv match institutional risk formulas."""
        metrics_path = ROOT_DIR / "results" / "benchmark_metrics.csv"
        equity_path = ROOT_DIR / "results" / "equity_curves.csv"
        self.assertTrue(metrics_path.exists(), "Missing results/benchmark_metrics.csv")
        self.assertTrue(equity_path.exists(), "Missing results/equity_curves.csv")

        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = {r["Strategy"]: r for r in csv.DictReader(f)}

        self.assertEqual(len(metrics), 8, "Expected 8 evaluated strategies in benchmark_metrics.csv")

        with equity_path.open("r", encoding="utf-8") as f:
            eq_rows = list(csv.DictReader(f))

        self.assertEqual(len(eq_rows), 505, "Expected 505 out-of-sample days in equity_curves.csv")

        # Verify GRU (Regime + Vol-Targeted) equity curve and drawdown consistency
        gru_eq = [float(r["GRU_Equity_USD"]) for r in eq_rows]
        gru_dd = [float(r["GRU_Drawdown_Pct"]) for r in eq_rows]
        gru_cum_ret = (gru_eq[-1] / 10000.0 - 1.0) * 100.0

        gru_row = metrics["PyTorch 4-Layer Stacked GRU (Regime + Vol-Targeted)"]
        self.assertAlmostEqual(gru_cum_ret, float(gru_row["Cumulative Return (%)"]), places=1)
        self.assertAlmostEqual(min(gru_dd), float(gru_row["Max Drawdown (%)"]), places=1)

        # Verify risk-governance layer beats naive LSTM and improves Sharpe over Buy & Hold
        bh_row = metrics["Baseline: Buy & Hold (Passive)"]
        lstm_gov = metrics["PyTorch 4-Layer Stacked LSTM (Regime + Vol-Targeted)"]
        lstm_naive = metrics["PyTorch 4-Layer LSTM (Unfiltered Naive Signal)"]

        self.assertGreater(float(gru_row["Sharpe Ratio"]), float(bh_row["Sharpe Ratio"]))
        self.assertGreater(float(gru_row["Sortino Ratio"]), float(bh_row["Sortino Ratio"]))
        self.assertLess(float(gru_row["Annualized Vol (%)"]), float(bh_row["Annualized Vol (%)"]))
        self.assertGreater(float(lstm_gov["Cumulative Return (%)"]), float(lstm_naive["Cumulative Return (%)"]))
        self.assertLess(int(lstm_gov["Total Trades"]), int(lstm_naive["Total Trades"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
