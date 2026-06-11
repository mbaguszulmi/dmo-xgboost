"""Data download, feature engineering, target construction, and splitting."""

import time

import numpy as np
import pandas as pd
import yfinance as yf

from .config import DATA_YEARS, HORIZON, TRAIN_RATIO
from .indicators import INDICATOR_BUILDERS

RETURN_LAGS = [1, 2, 3, 5, 10, 21]

FETCH_ATTEMPTS = 3
FETCH_BACKOFF_SECONDS = [2, 5]


def download_history(symbol: str) -> pd.DataFrame:
    """Last DATA_YEARS years of daily OHLCV (auto-adjusted) for a symbol.

    Yahoo Finance returns an empty DataFrame (without raising) on rate limits
    or transient errors, so empty responses are retried with a short backoff.
    """
    for attempt in range(FETCH_ATTEMPTS):
        df = yf.Ticker(symbol).history(period=f"{DATA_YEARS}y", auto_adjust=True)
        if not df.empty:
            break
        if attempt < FETCH_ATTEMPTS - 1:
            delay = FETCH_BACKOFF_SECONDS[attempt]
            print(f"  No data for '{symbol}' (attempt {attempt + 1}/"
                  f"{FETCH_ATTEMPTS}), retrying in {delay}s...")
            time.sleep(delay)
    else:
        raise ValueError(
            f"No data returned for '{symbol}' after {FETCH_ATTEMPTS} attempts — "
            "the symbol may be invalid, or Yahoo Finance may be rate-limiting; "
            "try again in a minute."
        )
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def build_features(history: pd.DataFrame, selected_indicators: list[str]) -> pd.DataFrame:
    """Scale-free feature matrix: base lags/volume plus selected indicators."""
    close = history["Close"]
    feats = pd.DataFrame(index=history.index)

    # Base features: lagged returns, rolling volatility, relative volume.
    ret1 = close.pct_change()
    for lag in RETURN_LAGS:
        feats[f"ret_{lag}d"] = close.pct_change(lag)
    feats["volatility_21d"] = ret1.rolling(21).std()
    vol_ma = history["Volume"].rolling(21).mean()
    feats["rel_volume"] = history["Volume"] / vol_ma.replace(0, np.nan)

    for name in selected_indicators:
        INDICATOR_BUILDERS[name](feats, close)

    return feats


def build_targets(close: pd.Series) -> pd.DataFrame:
    """Direct multi-output targets: return from t's close to each of the next
    HORIZON closes. Rows near the end of the series have NaN columns."""
    cols = {f"y_{h}": close.shift(-h) / close - 1.0 for h in range(1, HORIZON + 1)}
    return pd.DataFrame(cols, index=close.index)


def prepare_dataset(symbol: str, selected_indicators: list[str]):
    """Returns (history, X, Y) aligned on dates with complete features.

    Y rows may contain NaNs near the end (insufficient future data); callers
    decide how to handle them (training drops them, forecasting ignores Y).
    """
    history = download_history(symbol)
    X = build_features(history, selected_indicators)
    Y = build_targets(history["Close"])
    valid = X.dropna().index
    return history.loc[valid], X.loc[valid], Y.loc[valid]


def chronological_split(X: pd.DataFrame, Y: pd.DataFrame):
    """70/30 chronological split.

    The last HORIZON rows of the train window are dropped: their targets reach
    into the test period and would leak it into training. Test rows keep only
    complete targets so metrics cover all 240 steps.
    """
    split = int(len(X) * TRAIN_RATIO)
    X_train, Y_train = X.iloc[: max(split - HORIZON, 0)], Y.iloc[: max(split - HORIZON, 0)]
    X_test, Y_test = X.iloc[split:], Y.iloc[split:]

    train_mask = Y_train.notna().all(axis=1)
    test_mask = Y_test.notna().all(axis=1)
    X_train, Y_train = X_train[train_mask], Y_train[train_mask]
    X_test, Y_test = X_test[test_mask], Y_test[test_mask]

    if len(X_train) < 100 or len(X_test) < 1:
        raise ValueError(
            f"Not enough data after split (train={len(X_train)}, test={len(X_test)}); "
            "the symbol may have too little history for a 240-step horizon."
        )
    return X_train, Y_train, X_test, Y_test
