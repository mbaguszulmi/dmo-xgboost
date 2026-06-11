# PLAN.md: Stock Forecasting System Using Direct-Multi Output XGBoost

This project aims to build a multi-step stock closing price prediction system (240 trading days ahead) using the XGBoost algorithm with a Direct-Multi Output approach. It features comprehensive backtest simulations and performance visualizations.

## 1. Technology Stack & Libraries

- **Language**: Python 3.x
- **Data Source**: `yfinance`
- **Core Model**: `xgboost` (XGBRegressor) wrapped within `sklearn.multioutput.MultiOutputRegressor`
- **Feature Engineering**: `pandas`, `numpy`, `ta` (or manual calculation for MA, RSI, MACD, RPO)
- **Storage**: `joblib` (for model weight objects) & `json` (for model metadata)
- **Visualization**: `matplotlib`
- **Configuration**: `python-dotenv` (.env)

## 2. Configuration Structure (.env)

The `.env` file is used to define the list of supported stocks for the system.
Code snippet

```
# List of stock symbols (comma-separated)
# Example for global and Indonesian (IDX) stocks
TARGET_STOCKS=BBCA.JK,TLKM.JK,AAPL,MSFT
```

## 3. Data Pipeline & Feature Engineering

- **Data Range**: The last 10 years from the current execution date.
- **Data Splitting (Chronological)**:
    - **Training Set**: The first 7 years (~70% of the initial data chronologically).
    - **Testing Set**: The last 3 years (~30% of the recent data chronologically).
- **Dynamic Features (Optional during Training)**:
    - **Moving Average (MA)**: SMA_20 and SMA_50.
    - **RSI**: Relative Strength Index (14-period).
    - **MACD**: Moving Average Convergence Divergence (Line & Signal).
    - **RPO**: Relative Price Oscillator.
- **Target Label (Direct-Multi Output)**:
    - Construct a target matrix consisting of 240 forward-looking columns (step 1 to 240).
## 4. Model Architecture & Storage

- **Model**: One unique `MultiOutputRegressor(XGBRegressor())` per stock symbol.
- **Model Weight Storage**: Saved in `.joblib` format (Example: `models/AAPL_model.joblib`).
- **Metadata Storage (`models_info.json`)**:

JSON

```
{
  "AAPL": {
    "model_path": "models/AAPL_model.joblib",
    "training_date": "2026-06-11 14:05:00",
    "selected_features": ["RSI", "RPO"],
    "parameters": {
      "n_estimators": 100,
      "max_depth": 6,
      "learning_rate": 0.1
    },
    "metrics": {
      "MAE": 2.34,
      "MAPE": 0.015,
      "MDA": 0.62
    }
  }
}
```

## 5. Evaluation Metrics

- **MAE (Mean Absolute Error)**: Measures the average absolute error in nominal terms.
- **MAPE (Mean Absolute Percentage Error)**: Measures the prediction error as a percentage.
- **MDA (Mean Directional Accuracy)**: Measures the accuracy of the predicted price direction (whether the model correctly anticipates an upward or downward movement at each step).

## 6. System Features

The program runs via an interactive CLI menu with the following main options:

### Option 1: Model Training
1. Read the list of supported stocks from `.env`.
2. The user selects a specific stock to train (or chooses a _train all_ option).
3. The user selects the additional technical indicators to activate (MA, RSI, MACD, RPO) via a multi-select prompt.
4. Execute the training process on the first 7 years of data and test on the remaining 3 years.
5. Calculate MAE, MAPE, and MDA metrics on the testing set.
6. Save the model (`.joblib`) and update `models_info.json`.
7. **Training Chart**: Generates and saves a combined chart (Real Data vs. Prediction over the Testing period) that displays the metric values within the image. Save directory: `plots/training/`.

### Option 2: Model Evaluation (Load Model)
The system displays a list of available trained models from `models_info.json`. After selecting a model, the user chooses a sub-feature:

- **2.1. Backtest**
    - Runs a simulated trading execution over the last _n_ years using historical data.
    - **Parameters**: `time_window`: Default 1 month (~21 trading days) & `initial_capital`: Default 1000.
    - **Trading Logic**: If the projected price at the end of the window is predicted to rise > 1% compared to the current price, execute a **BUY** (all-in). The status changes to _Holding_. If the price is predicted to fall by the end of the window, execute a **SELL** to secure capital (status reverts to _Idle_).
    - **Backtest Charts**: Saves a single image containing 2 subplots. Top subplot: Real vs. Predicted price per step, displaying the MAPE value and Total Windows. Bottom subplot: Portfolio growth curve, with loss areas shaded in red, displaying Return, Win Rate, Total Trades, and Max Drawdown metrics. Save directory: `plots/backtest/`.
        
- **2.2. Forecast**
    - Predicts the closing prices for the next 240 steps (~1 trading year) ahead from the latest available data point.
    - **Forecast Chart**: Saves a single chart displaying the last 5 months of historical real data seamlessly connected to the 240-step forecast projection curve. Save directory: `plots/forecast/`.

## 7. Chart File Naming Standard

Every generated chart is automatically saved to its respective folder using the following naming convention: `[Stock]_YYYY-MM-DD_[HH:mm:ss].png` _(Example: `AAPL_2026-06-11_14:07:30.png`)_
