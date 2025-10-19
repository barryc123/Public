# 📈 Algorithmic Trading for Mean Reversion & Trend Following

This project explores the design, backtesting, optimization, and live deployment of **algorithmic trading strategies** based on **Mean Reversion** and **Trend Following** principles.  
The work applies quantitative methods to cryptocurrency markets using **Python**, the **Alpaca API**, and **Backtesting.py**.

---

## 🧠 Overview

Algorithmic trading enables systematic decision-making by combining **quantitative analysis** and **automated execution**.  
This project develops two proprietary strategy frameworks:

1. **Mean Reversion** — assumes prices revert to their long-term mean after extreme deviations.  
2. **Trend Following** — aims to capture price momentum in either direction.

Cryptocurrency markets (BCHUSD, ETHUSD, USDTUSD) were selected for their **high volatility** and **liquidity**, making them ideal for testing these approaches.

Key objectives:
- Develop and backtest multiple algorithmic strategies.
- Optimize parameters to maximize **Return-to-Drawdown ratio**.
- Deploy the best-performing strategies in **live trading environments** using Alpaca’s API.
- Evaluate results through systematic backtesting and statistical analysis.

---

## 🧩 Project Structure

```
AlgoTrading/
│
├── .env                                 # Environment variables (API keys, credentials)
│
├── BCHUSD_2023.csv                      # Alpaca OHLC data (5-min bars)
├── BCHUSD_2024.csv
├── ETHUSD_2023.csv
├── ETHUSD_2024.csv
├── USDTUSD_2023.csv
├── USDTUSD_2024.csv
├── DTB3.csv                             # 3-Month US Treasury Bill rates (FRED)
│
├── opt_results_2023-7-1_2023-12-31.csv  # Optimized backtest results
│
├── P1AGetData.ipynb                     # Data collection via Alpaca API
├── P1BTechIndicatorsPlots.ipynb         # Technical indicators visualizations
├── P1CMeanReversion.ipynb               # Stationarity & ADF testing
├── P1DAnalyseOptimisedBacktests.ipynb   # Optimization analysis
├── P1E2backtesting_strats.py            # Backtesting and optimization scripts
├── P1EStrategies.py                     # Backtesting strategy definitions
├── P1Ftechnical_indicators.py           # Technical indicator functions (SMA, EMA, RSI, ADX, MACD)
│
├── P2Amean_reversion_live_trading.py    # Live trading script: EMA + RSI
├── P2Btrend_follow_live_trading.py      # Live trading script: MACD + ADX
├── P2Cmean_reversion_logs.log           # Log: Mean Reversion strategy
├── P2Dtrend_follow_logs.log             # Log: Trend Following strategy
│
├── P3TradesAnalysis.ipynb               # 2024 live backtest performance analysis
│
├── ProjectSummary.pdf                   # Full CQF project report
└── README.md                            # This file
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- Dependencies:
  ```bash
  pip install backtesting pandas numpy matplotlib alpaca-trade-api python-decouple
  ```

### Configuration
1. Create a `.env` file in the project root:
   ```bash
   ALPACA_MEAN_KEY=your_api_key
   ALPACA_MEAN_SECRET=your_secret
   ALPACA_TREND_KEY=your_api_key
   ALPACA_TREND_SECRET=your_secret
   ```
2. Ensure all `.csv` data files are in the working directory.
3. Modify time intervals and parameters in scripts as required.

---

## 📊 Strategy Summaries

### 1. Mean Reversion — *EMA + RSI*

- **Concept**: Buy when price < EMA and RSI < lower threshold.  
- **Exit**: RSI > upper threshold or stop-loss / take-profit triggered.
- **Optimized Parameters (BCHUSD, 30-min data):**
  - `lower_rsi_band`: 30  
  - `upper_rsi_band`: 80  
  - `rsi_window`: 10  
  - `ema_window`: 70  
- **Results**:
  - Return-to-Drawdown Ratio: **4.34**
  - Win Rate: **90%**
  - Sharpe Ratio: **1.0**
  - Annual Return: **0.55%**

📁 Code: `P2Amean_reversion_live_trading.py`

---

### 2. Trend Following — *MACD + ADX*

- **Concept**: Buy when MACD crosses above signal, +DI > -DI, and ADX > 25.  
- **Exit**: ADX < 20 or stop-loss / take-profit reached.  
- **Optimized Parameters (BCHUSD, 30-min data):**
  - `macd_short_window`: 14  
  - `macd_long_window`: 28  
  - `adx_window`: 13  
- **Results**:
  - Annual Return: **-0.13%**
  - Return-to-Drawdown Ratio: **0.08**
  - Sharpe Ratio: **-4.39**
  - Total Trades: **178**
  - Max Drawdown Duration: **270 days**

📁 Code: `P2Btrend_follow_live_trading.py`

---

## 🧪 Backtesting & Optimization

Backtesting was performed using **Backtesting.py**, with each parameter optimized using **grid search** across defined ranges.  
Optimization aimed to **maximize the Return-to-Drawdown ratio**, balancing performance with risk.

**Metrics computed:**
- Annual Return (%)
- Win Rate (%)
- Sharpe Ratio
- Max Drawdown (%)
- Return-to-Drawdown Ratio
- Value-at-Risk (Parametric & Historical)
- Rolling Sharpe Ratio

📁 Files:
- `P1E2backtesting_strats.py`
- `P1DAnalyseOptimisedBacktests.ipynb`
- `opt_results_2023-7-1_2023-12-31.csv`

---

## 🔁 Live Trading Implementation

Live systems integrate directly with **Alpaca’s Crypto API** for real-time data and execution.  
Each bot:
- Pulls new 30-minute OHLC bars.
- Computes indicators (EMA, RSI, MACD, ADX).
- Manages open positions using stop-loss / take-profit levels.
- Logs activity to rotating files (`.log`).

Execution:
```bash
python P2Amean_reversion_live_trading.py
python P2Btrend_follow_live_trading.py
```

Example order executions are logged in:
- `P2Cmean_reversion_logs.log`
- `P2Dtrend_follow_logs.log`

---

## 📈 Performance Analysis

All results are visualized in `P3TradesAnalysis.ipynb`:
- **Equity Curves** (capital growth)
- **Drawdown Curves**
- **Rolling Sharpe Ratio**
- **1-Day & 10-Day Value-at-Risk (VaR)**

Example highlights (from report):
- *Mean Reversion*: steady capital preservation.  
- *Trend Following*: large trade volume but high drawdowns.  
- *Rolling VaR*: parametric VaR > historical VaR consistently throughout 2024.

---

## 🧾 Key Findings

- **Mean Reversion** effective in range-bound markets.  
- **Trend Following** beneficial during strong directional moves, but underperforms in volatile, sideways conditions.  
- **Data limitations**: only 30-min bars used (no tick data).  
- **Optimization risk**: overfitting possible; cross-validation recommended.  
- **Future Work**:
  - Combine mean reversion and trend following signals.
  - Diversify across multiple assets.
  - Integrate machine learning to dynamically adapt parameters.

---

## 🧰 Methods Implemented

| Method | Files | Description |
|--------|-------|-------------|
| SMA, EMA | `P1BTechIndicatorsPlots.ipynb`, `P1Ftechnical_indicators.py` | Moving averages |
| Bollinger Bands | `P1Ftechnical_indicators.py` | Volatility-based bands |
| RSI | `P1Ftechnical_indicators.py` | Momentum oscillator |
| MACD, ADX | `P1Ftechnical_indicators.py` | Trend strength indicators |
| ADF Test | `P1CMeanReversion.ipynb` | Stationarity testing |
| Sharpe Ratio, VaR | `P3TradesAnalysis.ipynb` | Risk metrics |

---

## 🧩 References

- Wilmott, P. (2006). *Paul Wilmott on Quantitative Finance*, Vol. 1, 2nd Ed.  
- FRED (Federal Reserve Bank of St. Louis): U.S. 3-Month Treasury Bill Rates.

---

## 👤 Author

**Barry Crosse**  
Certificate in Quantitative Finance (CQF) — Final Project  
© 2025 Barry Crosse. All rights reserved.
