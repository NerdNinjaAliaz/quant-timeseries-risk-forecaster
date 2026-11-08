# Institutional Risk Governance & Business Case Memo

**Project:** `quant-timeseries-risk-forecaster`  
**Author:** Ali Azizi (M.Sc. Biomedical Engineering, University of Tehran | AI Product Manager)  
**Context:** Portfolio Artifact for Master's in Business in Artificial Intelligence

## 1. Problem Statement: Why Academic Deep Learning Fails in Financial Production

In academic coursework (including my original 2021 study in [`bitcoin_predictor.ipynb`](https://colab.research.google.com/drive/17t40w1HfbbwlMBRGGUjjsXZ9YtEzZYRf?ouid=114192877993810143807)), recurrent neural networks (`LSTM`, `GRU`, `SimpleRNN`) are trained to minimize one-step-ahead price MSE over a 28-day sliding window (`[50, 60, 80, 120]` hidden units). When deployed directly into a commercial trading or treasury hedging workflow, three structural problems emerge:

1. **Data Leakage in Feature Normalization**: Calling `scaler.fit_transform(data_test)` independently on the test partition leaks the future price range (`$4,970` to `$63,503` in 2020–2021) into the model inputs. In production, the model only knows historical price bounds up to `t-1`.
2. **Fee & Slippage Erosion**: A model that predicts small positive/negative oscillations around the current price will flip between 100% Long and 100% Cash over 200 times in 505 days. At `10 bps` (`0.10%`) combined exchange fee and slippage per trade, 224 trades cost over `22%` of base capital in friction alone.
3. **Asymmetric Drawdown Risk**: Institutional allocators evaluate strategies on **Sharpe Ratio**, **Sortino Ratio**, **Calmar Ratio**, and **95% Conditional Value at Risk (CVaR)** rather than raw RMSE. Passive Bitcoin exposure during `2020-01-02` to `2021-05-20` suffered a **`-45.47%` Maximum Drawdown** and a **`-6.07%` Daily 95% CVaR**, exceeding the risk tolerance of corporate treasuries and balanced quant funds.

## 2. Solution Architecture: Three-Pillar Risk Governance

To transform the 4-layer `[50, 60, 80, 120]` **[PyTorch](https://github.com/pytorch/pytorch)** recurrent models into an investable decision-support system, we introduce three product controls in `src/backtest.py`:

* **Pillar 1 — Leak-Free Chronological Scaling (`src/data_pipeline.py`)**: All **[scikit-learn](https://github.com/scikit-learn/scikit-learn)** transformers (`MinMaxScaler`, `StandardScaler`, `PowerTransformer`) are fit strictly on the `2014-09-20` to `2019-12-31` training split (`1,929` days) and applied out-of-sample to `2020-01-02` through `2021-05-20` (`505` days).
* **Pillar 2 — Turnover-Buffered Regime Filter (`compute_signals_from_predictions`)**: Instead of exiting to cash on minor next-day pullbacks, the allocation engine combines the PyTorch RNN expected return threshold (`+20 bps`) with a macro trend buffer (`SMA 10 > SMA 30`). This cuts LSTM trade frequency from **224 trades** down to **58 trades**, increasing net cumulative return from **120.54%** to **335.64%**.
* **Pillar 3 — Inverse Volatility Targeting (`65%` Annualized Vol Cap)**: Position sizes scale dynamically between `0.50x` and `1.00x` (quantized in `0.25` steps to minimize rebalancing friction) based on trailing 14-day realized volatility. This compresses annualized portfolio volatility from **61.16%** (Passive Buy & Hold) down to **53.92%** (Stacked GRU) and improves the **Sharpe Ratio from `2.36` to `2.42`** (**Sortino Ratio from `3.70` to `3.90`**).

## 3. Executive Decision Matrix for AI Product Managers

When deciding whether to deploy a deep learning forecaster (`LSTM`/`GRU`) versus a deterministic rule-based baseline (`SMA 10/30 Crossover`), an AI Product Manager must weigh **infrastructure complexity** against **regime adaptability**:

* **When to Deploy Classical Rule-Based Baselines (`SMA 10/30`)**: In persistent, single-direction macro bull trends with low regime churn, simple moving average crossovers (`12` trades, **Sharpe `2.85`**, **Max Drawdown `-29.97%`**) offer zero GPU inference cost, complete explainability for compliance teams, and minimal transaction fee drag.
* **When to Deploy PyTorch `StackedGRU` / `StackedLSTM` + Vol-Targeting**: When incorporating multi-variate features (intraday High-Low spread, volume surges, on-chain telemetry, or macro sentiment) where linear moving averages lag sudden volatility spikes, the **Regime + Vol-Targeted Stacked GRU** provides superior downside tail protection (**Daily 95% VaR `-4.26%`** vs. Buy & Hold's **`-4.79%`**) while capturing **409.29%** cumulative upside.
