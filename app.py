"""Streamlit Executive Risk & Portfolio Backtesting Dashboard.

Launch locally:
    streamlit run app.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from run_experiment import train_and_predict
from src.backtest import (
    compute_baseline_signals,
    compute_signals_from_predictions,
    evaluate_portfolio_strategy,
)
from src.data_pipeline import load_and_prepare_data


st.set_page_config(
    page_title="Quant Time-Series Risk Forecaster | Ali Azizi",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Quantitative Time-Series Risk Forecaster & Portfolio Backtesting Engine")
st.markdown(
    """
    **Author:** Ali Azizi (M.Sc. Biomedical Engineering, University of Tehran | AI Product Manager)  
    **Executive Summary:** Upgrades a 4-layer stacked Recurrent Neural Network (`[50, 60, 80, 120]` hidden units with `[0.2, 0.3, 0.4, 0.5]` progressive dropout) into an institutional-grade **PyTorch** financial risk and portfolio backtesting engine. Compares deep learning forecasts against passive and rule-based quantitative baselines using **Sharpe Ratio**, **Sortino Ratio**, **Maximum Drawdown**, and **95% Value at Risk (VaR)** net of transaction costs.
    """
)

with st.sidebar:
    st.header("⚙️ Model & Risk Governance")
    architecture = st.selectbox(
        "PyTorch Architecture",
        options=["GRU", "LSTM", "RNN", "MLP"],
        index=0,
        help="4-layer stacked [50, 60, 80, 120] architecture ported from original study.",
    )
    scaler_name = st.selectbox(
        "Feature Scaling Pipeline (Leak-Free)",
        options=["minmax", "standard", "power"],
        index=0,
        help="Fit strictly on pre-2020 training data to prevent lookahead bias.",
    )
    use_dropout = st.checkbox("Enable Progressive Dropout [0.2–0.5]", value=True)
    vol_target = st.checkbox("Enable Volatility-Targeted Position Sizing", value=True)
    epochs = st.slider("Training Epochs", min_value=5, max_value=30, value=12, step=1)
    threshold_bps = st.slider("Minimum Signal Edge Threshold (bps)", min_value=0, max_value=100, value=20, step=5)
    fee_bps = st.slider("Transaction Cost & Slippage (bps/trade)", min_value=0, max_value=50, value=10, step=5)
    initial_capital = st.number_input("Initial Portfolio Capital ($ USD)", min_value=1000.0, value=10000.0, step=1000.0)

base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "data" / "BTC-USD.csv"

prepared = load_and_prepare_data(
    csv_path=csv_path,
    split_date="2020-01-01",
    lookback=28,
    scaler_name=scaler_name,
)

with st.spinner(f"Training PyTorch 4-Layer {architecture} & running out-of-sample backtest (2020–2021)..."):
    scaled_preds = train_and_predict(
        architecture=architecture,
        X_train=prepared.X_train,
        y_train=prepared.y_train,
        X_test=prepared.X_test,
        epochs=epochs,
        batch_size=50,
        use_dropout=use_dropout,
    )
    usd_preds = prepared.inverse_transform_target(scaled_preds)
    dl_signals = compute_signals_from_predictions(
        actual_close=prepared.test_close,
        predicted_close=usd_preds,
        threshold_bps=float(threshold_bps),
        volatility_target=vol_target,
    )
    dl_result = evaluate_portfolio_strategy(
        strategy_name=f"PyTorch 4-Layer {architecture} (Active)",
        actual_close=prepared.test_close,
        positions=dl_signals,
        predicted_close=usd_preds,
        initial_capital=float(initial_capital),
        fee_bps=float(fee_bps),
    )

    baselines = compute_baseline_signals(prepared.test_close)
    all_results = [dl_result]
    for b_name, b_sig in baselines.items():
        all_results.append(
            evaluate_portfolio_strategy(
                strategy_name=b_name,
                actual_close=prepared.test_close,
                positions=b_sig,
                predicted_close=None,
                initial_capital=float(initial_capital),
                fee_bps=float(fee_bps),
            )
        )

bh_result = all_results[1]

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Sharpe Ratio (Annualized)",
    f"{dl_result.sharpe_ratio:.2f}",
    delta=f"{dl_result.sharpe_ratio - bh_result.sharpe_ratio:+.2f} vs Buy & Hold",
)
col2.metric(
    "Maximum Drawdown",
    f"{dl_result.max_drawdown_pct:.2f}%",
    delta=f"{dl_result.max_drawdown_pct - bh_result.max_drawdown_pct:+.2f}% tail-risk reduction",
)
col3.metric(
    "Sortino Ratio",
    f"{dl_result.sortino_ratio:.2f}",
    delta=f"{dl_result.sortino_ratio - bh_result.sortino_ratio:+.2f} vs Buy & Hold",
)
col4.metric(
    "Daily 95% VaR / CVaR",
    f"{dl_result.var_95_pct:.2f}% / {dl_result.cvar_95_pct:.2f}%",
    delta=f"Calmar: {dl_result.calmar_ratio:.2f}",
)

st.subheader("1. Institutional Risk & Performance Scorecard (Out-of-Sample: 2020-01-02 to 2021-05-20)")
scorecard_df = pd.DataFrame(
    [
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
            "Trades": r.total_trades,
        }
        for r in all_results
    ]
)
st.dataframe(scorecard_df, use_container_width=True)

st.subheader("2. Out-of-Sample Portfolio Equity Curves ($10,000 Initial Capital)")
equity_chart_df = pd.DataFrame(
    {r.strategy_name: r.equity_curve for r in all_results},
    index=pd.to_datetime(prepared.test_dates),
)
st.line_chart(equity_chart_df)

st.subheader("3. Underwater Drawdown Profile (%) — Capital Preservation Audit")
dd_chart_df = pd.DataFrame(
    {r.strategy_name: r.drawdown_curve for r in all_results},
    index=pd.to_datetime(prepared.test_dates),
)
st.area_chart(dd_chart_df)

st.subheader(f"4. {architecture} Next-Day Forecast vs. Realized BTC-USD Price")
forecast_df = pd.DataFrame(
    {
        "Realized BTC-USD Close": prepared.test_close,
        f"PyTorch {architecture} Predicted Close": usd_preds,
    },
    index=pd.to_datetime(prepared.test_dates),
)
st.line_chart(forecast_df)
