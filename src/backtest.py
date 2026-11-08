"""Financial Risk & Portfolio Backtesting Engine.

Evaluates deep learning price forecasts (LSTM, GRU, RNN, MLP) against classical
quantitative baselines (Buy & Hold, SMA 10/30 Crossover, Time-Series Momentum)
using institutional portfolio risk metrics:
- Annualized Return (CAGR) & Cumulative Return
- Annualized Sharpe Ratio & Sortino Ratio
- Maximum Drawdown (Max DD) & Calmar Ratio
- Daily 95% Historical Value at Risk (VaR) & Conditional VaR (CVaR / Expected Shortfall)
- Transaction Cost & Slippage Accounting (bps per turnover)
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Summary metrics and daily equity series for a strategy."""

    strategy_name: str
    cumulative_return_pct: float
    cagr_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    var_95_pct: float
    cvar_95_pct: float
    win_rate_pct: float
    total_trades: int
    rmse_usd: float
    mae_usd: float
    equity_curve: np.ndarray
    drawdown_curve: np.ndarray
    daily_returns: np.ndarray
    positions: np.ndarray


def compute_signals_from_predictions(
    actual_close: np.ndarray,
    predicted_close: np.ndarray,
    threshold_bps: float = 20.0,
    volatility_target: bool = True,
    regime_filter: bool = True,
) -> np.ndarray:
    """Convert next-day price forecasts into risk-managed portfolio weights in [0, 1].

    Combines deep learning return forecasts with:
    1. Regime & turnover buffer (`regime_filter=True`): holds conviction during macro uptrends
       while exiting when both the RNN forecast and short-term trend turn negative (-35 bps),
       preventing excessive transaction fee drag.
    2. Inverse volatility targeting (`volatility_target=True`): scales exposure inversely to
       14-day realized volatility during extreme turbulence to compress Maximum Drawdown.
    """
    prev_close = np.roll(actual_close, 1)
    prev_close[0] = actual_close[0]

    expected_return = (predicted_close - prev_close) / np.maximum(prev_close, 1e-8)
    min_edge = threshold_bps / 10000.0

    if not regime_filter:
        raw_signal = np.where(expected_return > min_edge, 1.0, 0.0)
    else:
        close_s = pd.Series(actual_close)
        sma_10 = close_s.rolling(10, min_periods=1).mean().shift(1).bfill().values
        sma_30 = close_s.rolling(30, min_periods=1).mean().shift(1).bfill().values
        raw_signal = np.zeros_like(actual_close, dtype=np.float64)
        state = 1.0
        for i in range(len(actual_close)):
            if expected_return[i] > min_edge or (sma_10[i] > sma_30[i] and expected_return[i] > -0.0045):
                state = 1.0
            elif expected_return[i] < -0.0035 and sma_10[i] <= sma_30[i]:
                state = 0.0
            raw_signal[i] = state

    if not volatility_target:
        return raw_signal

    realized_ret = pd.Series(actual_close).pct_change().fillna(0.0)
    rolling_vol = realized_ret.rolling(14, min_periods=3).std().fillna(0.03) * np.sqrt(365.0)
    target_vol = 0.65
    # Quantize vol scalar to 0.25 increments to avoid micro-rebalancing fee drag
    vol_scalar = np.clip(np.round((target_vol / np.maximum(rolling_vol.values, 0.15)) * 4.0) / 4.0, 0.50, 1.0)
    return np.round(raw_signal * vol_scalar, 4)


def compute_baseline_signals(actual_close: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute classical quantitative baseline strategies over the same test window."""
    close_series = pd.Series(actual_close)

    buy_and_hold = np.ones_like(actual_close, dtype=np.float64)

    sma_fast = close_series.rolling(10, min_periods=1).mean()
    sma_slow = close_series.rolling(30, min_periods=1).mean()
    sma_crossover = np.where(sma_fast.values > sma_slow.values, 1.0, 0.0)
    sma_crossover = np.roll(sma_crossover, 1)
    sma_crossover[0] = 1.0

    mom_14 = close_series.pct_change(14).fillna(0.0).values
    ts_momentum = np.where(mom_14 > 0.0, 1.0, 0.0)
    ts_momentum = np.roll(ts_momentum, 1)
    ts_momentum[0] = 1.0

    return {
        "Baseline: Buy & Hold (Passive)": buy_and_hold,
        "Baseline: SMA 10/30 Crossover": sma_crossover,
        "Baseline: 14D Momentum": ts_momentum,
    }


def evaluate_portfolio_strategy(
    strategy_name: str,
    actual_close: np.ndarray,
    positions: np.ndarray,
    predicted_close: np.ndarray = None,
    initial_capital: float = 10000.0,
    fee_bps: float = 10.0,
    risk_free_rate_annual: float = 0.02,
    trading_days_per_year: int = 365,
) -> BacktestResult:
    """Run vector backtest with transaction costs and compute institutional risk metrics."""
    asset_returns = np.zeros_like(actual_close, dtype=np.float64)
    asset_returns[1:] = (actual_close[1:] - actual_close[:-1]) / np.maximum(actual_close[:-1], 1e-8)

    turnover = np.abs(np.diff(positions, prepend=0.0))
    cost_rate = fee_bps / 10000.0
    transaction_costs = turnover * cost_rate

    strategy_returns = (positions * asset_returns) - transaction_costs
    equity_curve = initial_capital * np.cumprod(1.0 + strategy_returns)

    running_peak = np.maximum.accumulate(equity_curve)
    drawdown_curve = (equity_curve - running_peak) / np.maximum(running_peak, 1e-8)
    max_drawdown = float(np.min(drawdown_curve))

    n_days = max(len(actual_close), 1)
    total_return = (equity_curve[-1] / initial_capital) - 1.0
    years = n_days / float(trading_days_per_year)
    cagr = (equity_curve[-1] / initial_capital) ** (1.0 / max(years, 1e-4)) - 1.0

    ann_vol = float(np.std(strategy_returns, ddof=1) * np.sqrt(trading_days_per_year))
    daily_rf = risk_free_rate_annual / float(trading_days_per_year)
    excess_returns = strategy_returns - daily_rf

    sharpe = (
        float(np.mean(excess_returns) / np.std(strategy_returns, ddof=1) * np.sqrt(trading_days_per_year))
        if np.std(strategy_returns, ddof=1) > 1e-10
        else 0.0
    )

    downside_returns = np.minimum(excess_returns, 0.0)
    downside_std = float(np.sqrt(np.mean(downside_returns**2)) * np.sqrt(trading_days_per_year))
    sortino = (
        float((np.mean(excess_returns) * trading_days_per_year) / downside_std)
        if downside_std > 1e-10
        else 0.0
    )

    calmar = float(cagr / abs(max_drawdown)) if abs(max_drawdown) > 1e-10 else 0.0

    var_95 = float(np.percentile(strategy_returns, 5))
    tail_losses = strategy_returns[strategy_returns <= var_95]
    cvar_95 = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_95

    active_days = strategy_returns[positions > 0.01]
    win_rate = float(np.mean(active_days > 0.0)) if len(active_days) > 0 else 0.0
    total_trades = int(np.sum(turnover > 0.05))

    if predicted_close is not None:
        rmse = float(np.sqrt(np.mean((actual_close - predicted_close) ** 2)))
        mae = float(np.mean(np.abs(actual_close - predicted_close)))
    else:
        rmse = np.nan
        mae = np.nan

    return BacktestResult(
        strategy_name=strategy_name,
        cumulative_return_pct=round(total_return * 100.0, 2),
        cagr_pct=round(cagr * 100.0, 2),
        annualized_volatility_pct=round(ann_vol * 100.0, 2),
        sharpe_ratio=round(sharpe, 2),
        sortino_ratio=round(sortino, 2),
        max_drawdown_pct=round(max_drawdown * 100.0, 2),
        calmar_ratio=round(calmar, 2),
        var_95_pct=round(var_95 * 100.0, 2),
        cvar_95_pct=round(cvar_95 * 100.0, 2),
        win_rate_pct=round(win_rate * 100.0, 2),
        total_trades=total_trades,
        rmse_usd=round(rmse, 2) if not np.isnan(rmse) else np.nan,
        mae_usd=round(mae, 2) if not np.isnan(mae) else np.nan,
        equity_curve=np.round(equity_curve, 2),
        drawdown_curve=np.round(drawdown_curve * 100.0, 2),
        daily_returns=np.round(strategy_returns * 100.0, 4),
        positions=positions,
    )
