# Stock Forecasting — Direct-Multi Output XGBoost

Multi-step stock closing price prediction (240 trading days ahead, ~1 trading
year) using `MultiOutputRegressor(XGBRegressor)` — one direct model per
forecast horizon, one bundle per stock. Includes backtest simulation and chart
generation. See `PLAN.md` for the original design.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then edit TARGET_STOCKS as needed
```

## Usage

```bash
python main.py
```

Interactive menu:

1. **Model Training** — pick a stock from `.env` (or train all), toggle
   optional indicators (MA, RSI, MACD, RPO), train on the first ~7 of 10
   years, evaluate MAE/MAPE/MDA on the last ~3 years, save the model to
   `models/` and a Real-vs-Prediction chart to `plots/training/`.
2. **Model Evaluation** — pick a trained model from `models_info.json`, then:
   - **Backtest** — windowed trading simulation (defaults: 21-day window,
     1000 capital, 0.15% fee per trade side). BUY all-in when the predicted
     window-end rise exceeds 1%; SELL when a fall is predicted; hold in the
     0–1% dead zone. Chart with price comparison + portfolio curve goes to
     `plots/backtest/`.
   - **Forecast** — 240-step projection from the latest close, charted against
     the last 5 months of history in `plots/forecast/`.

## Design notes (deviations from PLAN.md, and why)

- **Targets are returns, not raw prices.** Tree models cannot extrapolate
  beyond the target range seen in training, so raw-price targets flatline on
  trending stocks. The model predicts `close[t+h]/close[t] - 1` for
  h = 1..240; prices are reconstructed for metrics and charts.
- **Features are scale-free**: lagged returns, 21-day volatility, relative
  volume, plus the selected indicators (MA and MACD normalized by price).
- **Leakage guards**: the last 240 training rows are dropped (their targets
  reach into the test period) and backtests are capped at 3 years (older data
  is the model's training set).
- **Fees**: each trade side costs a configurable fee (default 0.15%) for a
  more realistic backtest; set 0 for frictionless results.
- **Models are saved per indicator combination** — e.g. TSLA trained with
  MA+RPO becomes `models/TSLA_MA-RPO_model.joblib` (indicators in canonical
  MA/RSI/MACD/RPO order), so variants of the same stock coexist and all appear
  in the evaluation menu.
- **Selected indicators are drawn on every chart**: SMA 20/50 overlay on the
  price axis; RSI, MACD, and RPO get their own panels below. On forecast
  charts they cover only the historical segment.
- **Backtests can span up to 10 years**, but periods before the ~3-year test
  window overlap the model's training data — the CLI warns about the
  look-ahead bias (results there will look unrealistically good).
- Chart filenames follow `[SYMBOL]_[INDICATORS]_TIMESTAMP` with `HH-MM-SS`
  (colons are invalid on some filesystems):
  `plots/<feature>/TSLA_MA-RPO_YYYY-MM-DD_HH-MM-SS.png`.
