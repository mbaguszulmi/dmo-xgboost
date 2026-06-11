"""Interactive CLI for the Direct-Multi Output XGBoost stock forecasting system."""

import sys

import pandas as pd
import questionary

from src.config import (
    DEFAULT_FEE_RATE,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_TIME_WINDOW,
    INDICATORS,
    MAX_BACKTEST_YEARS,
    TRAIN_RATIO,
    get_target_stocks,
    model_id,
)
from src.backtest import run_backtest
from src.data import build_features, download_history
from src.forecast import run_forecast
from src.model import load_model, load_models_info, train_symbol
from src.plotting import save_backtest_chart, save_forecast_chart, save_training_chart

TRAIN_ALL = "[ Train ALL stocks ]"
BACK = "← Back"


def menu_train() -> None:
    stocks = get_target_stocks()
    if not stocks:
        print("No stocks configured. Set TARGET_STOCKS in .env first.")
        return

    while True:
        choice = questionary.select(
            "Select a stock to train:", choices=[*stocks, TRAIN_ALL, BACK]
        ).ask()
        if choice is None or choice == BACK:
            return

        selected = questionary.checkbox(
            "Select additional technical indicators (space to toggle, "
            "Ctrl+C to go back):",
            choices=INDICATORS,
        ).ask()
        if selected is not None:
            break  # cancelled checkbox -> back to stock selection

    targets = stocks if choice == TRAIN_ALL else [choice]
    for symbol in targets:
        print(f"\nTraining {symbol} (indicators: {selected or 'none'})...")
        try:
            _, info, plot_data = train_symbol(symbol, selected)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        m = info["metrics"]
        print(f"  MAE={m['MAE']}  MAPE={m['MAPE']:.2%}  MDA={m['MDA']:.2%}")
        chart = save_training_chart(model_id(symbol, selected), plot_data, selected)
        print(f"  Model saved: {info['model_path']}")
        print(f"  Chart saved: {chart}")


def _ask_number(prompt: str, default, cast):
    answer = questionary.text(prompt, default=str(default)).ask()
    if answer is None:
        return None
    try:
        return cast(answer)
    except ValueError:
        print(f"Invalid number, using default {default}.")
        return cast(default)


def _model_choices(info: dict) -> list:
    """Selectable rows with metrics, ordered by MDA descending (best first)."""
    def mda(item):
        return item[1].get("metrics", {}).get("MDA", float("-inf"))

    choices = []
    for key, entry in sorted(info.items(), key=mda, reverse=True):
        m = entry.get("metrics", {})
        title = (f"{key:<24} MDA {m.get('MDA', 0):>7.2%}   "
                 f"MAPE {m.get('MAPE', 0):>7.2%}   MAE {m.get('MAE', 0):>10.4f}")
        choices.append(questionary.Choice(title=title, value=key))
    return [*choices, questionary.Choice(title=BACK, value=BACK)]


def menu_evaluate() -> None:
    info = load_models_info()
    if not info:
        print("No trained models found. Train a model first (option 1).")
        return

    while True:
        print(f"  {'Model':<24} {'MDA':>11}   {'MAPE':>12}   {'MAE':>14}")
        mid = questionary.select(
            "Select a trained model (best MDA first):", choices=_model_choices(info)
        ).ask()
        if mid is None or mid == BACK:
            return
        entry = info[mid]
        symbol = entry.get("symbol", mid)  # old entries were keyed by bare symbol
        print(f"Trained: {entry['training_date']} | indicators: "
              f"{entry['selected_features'] or 'none'} | metrics: {entry['metrics']}")
        model = load_model(mid)
        indicators = entry["selected_features"]

        while True:
            action = questionary.select(
                "Choose an evaluation feature:",
                choices=["Backtest", "Forecast", BACK],
            ).ask()
            if action is None or action == BACK:
                break  # back to model selection
            try:
                # Name charts by symbol+indicators even for old registry
                # entries that were keyed by bare symbol.
                _run_evaluation(action, model_id(symbol, indicators), symbol,
                                model, indicators)
            except Exception as e:
                print(f"  FAILED: {e}")


GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"


def _print_transactions(transactions: list[dict]) -> None:
    """Backtest transaction table; SELL rows colored green (gain) / red (loss)."""
    if not transactions:
        print("  No trades were executed in this backtest.")
        return
    print(f"\n  {'Date':<12} {'Action':<10} {'Price':>12} {'Shares':>12} "
          f"{'Value':>12} {'P/L':>12} {'P/L %':>9}")
    print("  " + "-" * 84)
    for t in transactions:
        row = (f"  {t['date'].strftime('%Y-%m-%d'):<12} {t['action']:<10} "
               f"{t['price']:>12.2f} {t['shares']:>12.4f} {t['value']:>12.2f}")
        if t["pl"] is None:
            print(row)
        else:
            color = GREEN if t["pl"] >= 0 else RED
            print(f"{row}{color}{t['pl']:>+12.2f}{t['pl_pct']:>+9.2%}{RESET}")
    print()


def _run_evaluation(action: str, mid: str, symbol: str, model,
                    indicators: list[str]) -> None:
    if action == "Forecast":
        print(f"Forecasting next 240 steps for {symbol}...")
        history_close, forecast = run_forecast(model, symbol, indicators)
        chart = save_forecast_chart(mid, history_close, forecast, indicators)
        last, end = history_close.iloc[-1], forecast.iloc[-1]
        print(f"  Last close: {last:.2f} -> predicted in 240 steps: {end:.2f} "
              f"({end / last - 1:+.2%})")
        print(f"  Chart saved: {chart}")
        return

    years = _ask_number(f"Backtest period in years (max {MAX_BACKTEST_YEARS}):",
                        MAX_BACKTEST_YEARS, int)
    window = _ask_number("Time window in trading days:", DEFAULT_TIME_WINDOW, int)
    capital = _ask_number("Initial capital:", DEFAULT_INITIAL_CAPITAL, float)
    fee = _ask_number("Fee rate per trade (0.0015 = 0.15%):", DEFAULT_FEE_RATE, float)
    if None in (years, window, capital, fee):
        return

    print(f"Running backtest for {symbol}...")
    history = download_history(symbol)
    X = build_features(history, indicators).dropna()

    test_start = X.index[int(len(X) * TRAIN_RATIO)]
    if X.index[-1] - pd.DateOffset(years=min(years, MAX_BACKTEST_YEARS)) < test_start:
        print(f"  WARNING: the model was trained on data before "
              f"{test_start.date()}; backtest results before that date overlap "
              "its training set (look-ahead bias) and will look optimistic.")

    result = run_backtest(model, history, X, years=years, time_window=window,
                          initial_capital=capital, fee_rate=fee)
    print(f"  Return: {result['return_pct']:+.2%} | Win rate: "
          f"{result['win_rate']:.2%} | Trades: {result['total_trades']} | "
          f"Max drawdown: {result['max_drawdown']:.2%}")
    _print_transactions(result["transactions"])
    chart = save_backtest_chart(mid, result, history["Close"], indicators)
    print(f"  Chart saved: {chart}")


def main() -> None:
    print("=== Stock Forecasting — Direct-Multi Output XGBoost ===")
    while True:
        action = questionary.select(
            "Main menu:",
            choices=["1. Model Training", "2. Model Evaluation (Load Model)", "Exit"],
        ).ask()
        if action is None or action == "Exit":
            print("Bye.")
            return
        if action.startswith("1"):
            menu_train()
        else:
            menu_evaluate()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
