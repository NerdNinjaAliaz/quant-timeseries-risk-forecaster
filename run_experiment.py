"""CLI Runner for Training PyTorch LSTM/GRU/RNN/MLP Models & Running Portfolio Backtests.

Usage:
    python3 run_experiment.py --epochs 15 --batch-size 50 --scaler minmax --fee-bps 10
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.backtest import (
    compute_baseline_signals,
    compute_signals_from_predictions,
    evaluate_portfolio_strategy,
)
from src.data_pipeline import load_and_prepare_data
from src.models import build_model


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_predict(
    architecture: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    epochs: int = 15,
    batch_size: int = 50,
    lr: float = 1e-3,
    use_dropout: bool = True,
    loss_type: str = "mse",
) -> np.ndarray:
    """Train a 4-layer PyTorch deep learning model and return scaled test predictions."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_split = int(len(X_train) * 0.9)
    X_tr, y_tr = X_train[:val_split], y_train[:val_split]

    dataset = TensorDataset(
        torch.from_numpy(X_tr).float(),
        torch.from_numpy(y_tr).float().unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = build_model(
        architecture=architecture,
        seq_len=X_train.shape[1],
        input_dim=X_train.shape[2],
        use_dropout=use_dropout,
    ).to(device)

    criterion = nn.MSELoss() if loss_type.lower() == "mse" else nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_tensor = torch.from_numpy(X_test).float().to(device)
        scaled_preds = model(test_tensor).cpu().numpy().flatten()

    return scaled_preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Quantitative Time-Series Risk Forecaster Benchmark")
    parser.add_argument("--data-path", type=str, default="data/BTC-USD.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--scaler", type=str, default="minmax", choices=["minmax", "standard", "power"])
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / args.data_path
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    prepared = load_and_prepare_data(
        csv_path=csv_path,
        split_date="2020-01-01",
        lookback=28,
        scaler_name=args.scaler,
    )

    architectures = [
        ("PyTorch 4-Layer Stacked GRU (Vol-Targeted)", "GRU", True),
        ("PyTorch 4-Layer Stacked LSTM (Vol-Targeted)", "LSTM", True),
        ("PyTorch 4-Layer SimpleRNN (Vol-Targeted)", "RNN", True),
        ("PyTorch Temporal MLP (28D Flattened)", "MLP", True),
    ]

    all_results = []
    equity_df = pd.DataFrame({"Date": prepared.test_dates, "BTC_Close_USD": prepared.test_close})

    for label, arch, drop in architectures:
        scaled_preds = train_and_predict(
            architecture=arch,
            X_train=prepared.X_train,
            y_train=prepared.y_train,
            X_test=prepared.X_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_dropout=drop,
        )
        usd_preds = prepared.inverse_transform_target(scaled_preds)
        signals = compute_signals_from_predictions(
            actual_close=prepared.test_close,
            predicted_close=usd_preds,
            threshold_bps=20.0,
            volatility_target=True,
        )
        res = evaluate_portfolio_strategy(
            strategy_name=label,
            actual_close=prepared.test_close,
            positions=signals,
            predicted_close=usd_preds,
            initial_capital=args.initial_capital,
            fee_bps=args.fee_bps,
        )
        all_results.append(res)
        equity_df[f"{arch}_Predicted_USD"] = np.round(usd_preds, 2)
        equity_df[f"{arch}_Equity_USD"] = res.equity_curve
        equity_df[f"{arch}_Drawdown_Pct"] = res.drawdown_curve

    baselines = compute_baseline_signals(prepared.test_close)
    for base_label, base_signals in baselines.items():
        res = evaluate_portfolio_strategy(
            strategy_name=base_label,
            actual_close=prepared.test_close,
            positions=base_signals,
            predicted_close=None,
            initial_capital=args.initial_capital,
            fee_bps=args.fee_bps,
        )
        all_results.append(res)
        col_key = base_label.replace("Baseline: ", "").replace(" ", "_").replace("&", "and").replace("/", "_")
        equity_df[f"{col_key}_Equity_USD"] = res.equity_curve
        equity_df[f"{col_key}_Drawdown_Pct"] = res.drawdown_curve

    summary_rows = []
    for r in all_results:
        summary_rows.append(
            {
                "Strategy": r.strategy_name,
                "Cumulative Return (%)": r.cumulative_return_pct,
                "CAGR (%)": r.cagr_pct,
                "Annualized Vol (%)": r.annualized_volatility_pct,
                "Sharpe Ratio": r.sharpe_ratio,
                "Sortino Ratio": r.sortino_ratio,
                "Max Drawdown (%)": r.max_drawdown_pct,
                "Calmar Ratio": r.calmar_ratio,
                "Daily VaR 95% (%)": r.var_95_pct,
                "Daily CVaR 95% (%)": r.cvar_95_pct,
                "Win Rate (%)": r.win_rate_pct,
                "Total Trades": r.total_trades,
                "Forecast RMSE ($)": r.rmse_usd,
                "Forecast MAE ($)": r.mae_usd,
            }
        )

    metrics_df = pd.DataFrame(summary_rows)
    metrics_df.to_csv(results_dir / "benchmark_metrics.csv", index=False)
    equity_df.to_csv(results_dir / "equity_curves.csv", index=False)

    print("\n=== QUANTITATIVE RISK & PORTFOLIO BACKTEST SCORECARD (Out-of-Sample: 2020-01-02 to 2021-05-20) ===")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
