"""Forecast the next HORIZON closing prices from the latest available data."""

import pandas as pd

from .config import HORIZON
from .data import build_features, download_history


def run_forecast(model, symbol: str, selected_indicators: list[str]):
    """Returns (history_close, forecast) where forecast is a price Series
    indexed by future business days."""
    history = download_history(symbol)
    X = build_features(history, selected_indicators).dropna()

    last_date = X.index[-1]
    last_close = history["Close"].loc[last_date]
    pred_returns = model.predict(X.iloc[[-1]].values)[0]

    future_dates = pd.bdate_range(start=last_date, periods=HORIZON + 1)[1:]
    forecast = pd.Series(last_close * (1.0 + pred_returns), index=future_dates,
                         name="forecast")
    return history["Close"].loc[:last_date], forecast
