"""Technical indicators, computed manually on a Close price series.

Price-scaled indicators (MA, MACD) are expressed relative to price so that
features stay scale-free; the model predicts returns, so features must not
carry absolute price levels.
"""

import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def add_ma(df: pd.DataFrame, close: pd.Series) -> None:
    """Close relative to SMA_20 / SMA_50 (e.g. +0.05 = 5% above the average)."""
    df["MA20_ratio"] = close / sma(close, 20) - 1.0
    df["MA50_ratio"] = close / sma(close, 50) - 1.0


def add_rsi(df: pd.DataFrame, close: pd.Series, period: int = 14) -> None:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100.0 - 100.0 / (1.0 + rs)


def add_macd(df: pd.DataFrame, close: pd.Series) -> None:
    """MACD line and signal line (12/26/9), normalized by price."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd_line / close
    df["MACD_signal"] = signal_line / close


def add_rpo(df: pd.DataFrame, close: pd.Series, fast: int = 10, slow: int = 30) -> None:
    """Relative Price Oscillator: % spread between fast and slow SMA."""
    fast_ma = sma(close, fast)
    slow_ma = sma(close, slow)
    df["RPO"] = (fast_ma - slow_ma) / slow_ma * 100.0


INDICATOR_BUILDERS = {
    "MA": add_ma,
    "RSI": add_rsi,
    "MACD": add_macd,
    "RPO": add_rpo,
}


def display_series(close: pd.Series, selected: list[str]) -> dict:
    """Conventional indicator values for charting (unlike the scale-free
    feature transforms). 'MA' maps to price-axis overlay lines; the other
    indicators map to {line_name: series} for their own panel."""
    out = {}
    if "MA" in selected:
        out["MA"] = {"SMA 20": sma(close, 20), "SMA 50": sma(close, 50)}
    if "RSI" in selected:
        tmp = pd.DataFrame(index=close.index)
        add_rsi(tmp, close)
        out["RSI"] = {"RSI": tmp["RSI"]}
    if "MACD" in selected:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        out["MACD"] = {"MACD": macd_line,
                       "Signal": macd_line.ewm(span=9, adjust=False).mean()}
    if "RPO" in selected:
        tmp = pd.DataFrame(index=close.index)
        add_rpo(tmp, close)
        out["RPO"] = {"RPO": tmp["RPO"]}
    return out
