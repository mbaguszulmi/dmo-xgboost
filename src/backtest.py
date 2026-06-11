"""Backtest simulation: windowed trading driven by model predictions.

Strategy per window of `time_window` trading days, evaluated at the window
start using the model's predicted return at horizon = time_window:
  - Idle  + predicted rise > BUY_THRESHOLD  -> BUY all-in (fee applied)
  - Holding + predicted fall (< 0)          -> SELL all (fee applied)
  - Holding + predicted change in [0, 1%]   -> keep holding (dead zone)
A position still open at the end is liquidated at the final close.
"""

import numpy as np
import pandas as pd

from .config import BUY_THRESHOLD, MAX_BACKTEST_YEARS


def run_backtest(model, history: pd.DataFrame, X: pd.DataFrame, *,
                 years: int, time_window: int, initial_capital: float,
                 fee_rate: float) -> dict:
    years = min(years, MAX_BACKTEST_YEARS)
    close = history["Close"]

    start_date = X.index[-1] - pd.DateOffset(years=years)
    bt_index = X.index[X.index >= start_date]
    # Each anchor needs a full window of future data after it.
    anchors = [a for a in bt_index[::time_window]
               if X.index.get_loc(a) + time_window < len(X)]
    if not anchors:
        raise ValueError("Backtest period too short for the chosen time window.")

    preds = model.predict(X.loc[anchors].values)[:, time_window - 1]
    anchor_pred = dict(zip(anchors, preds))

    cash = initial_capital
    shares = 0.0
    cost_basis = 0.0
    wins = 0
    trades = 0
    window_rows = []
    daily_values = []
    transactions = []

    def sell(date, price: float, action: str = "SELL"):
        nonlocal cash, shares, wins, trades
        sold_shares = shares
        cash = shares * price * (1.0 - fee_rate)
        shares = 0.0
        trades += 1
        wins += cash > cost_basis
        transactions.append({
            "date": date, "action": action, "price": price,
            "shares": sold_shares, "value": cash,
            "pl": cash - cost_basis, "pl_pct": cash / cost_basis - 1.0,
        })

    for date in bt_index:
        price = close.loc[date]
        pred_ret = anchor_pred.get(date)
        if pred_ret is not None:
            pos = X.index.get_loc(date)
            end_date = X.index[pos + time_window]
            window_rows.append((end_date, close.loc[end_date],
                                price * (1.0 + pred_ret)))
            if shares == 0.0 and pred_ret > BUY_THRESHOLD:
                cost_basis = cash
                shares = cash * (1.0 - fee_rate) / price
                cash = 0.0
                transactions.append({
                    "date": date, "action": "BUY", "price": price,
                    "shares": shares, "value": cost_basis,
                    "pl": None, "pl_pct": None,
                })
            elif shares > 0.0 and pred_ret < 0.0:
                sell(date, price)
        daily_values.append(cash + shares * price)

    if shares > 0.0:  # liquidate at the final close
        sell(bt_index[-1], close.loc[bt_index[-1]], action="SELL (end)")
        daily_values[-1] = cash

    portfolio = pd.Series(daily_values, index=bt_index, name="portfolio")

    windows = pd.DataFrame(window_rows, columns=["date", "actual", "predicted"])
    windows = windows.set_index("date")
    window_mape = float(np.mean(
        np.abs((windows["actual"] - windows["predicted"]) / windows["actual"])
    ))

    running_max = portfolio.cummax()
    max_drawdown = float(((portfolio - running_max) / running_max).min())

    final_value = float(portfolio.iloc[-1])
    return {
        "windows": windows,
        "portfolio": portfolio,
        "transactions": transactions,
        "initial_capital": initial_capital,
        "final_value": final_value,
        "return_pct": final_value / initial_capital - 1.0,
        "win_rate": wins / trades if trades else 0.0,
        "total_trades": trades,
        "max_drawdown": max_drawdown,
        "window_mape": window_mape,
        "years": years,
    }
