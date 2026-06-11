"""Chart generation. Every chart is saved as
[SYMBOL]_[INDICATORS]_YYYY-MM-DD_[HH-mm-ss].png in its feature's folder under
plots/ (the indicator part is omitted for models trained without indicators).

Selected indicators are drawn on every chart: SMA 20/50 overlay on the price
axis, while RSI / MACD / RPO each get their own panel below the main plots.
"""

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    HORIZON,
    PLOTS_BACKTEST_DIR,
    PLOTS_FORECAST_DIR,
    PLOTS_TRAINING_DIR,
    ensure_dirs,
)
from .indicators import display_series

PANEL_INDICATORS = ["RSI", "MACD", "RPO"]  # drawn below the price chart
PANEL_COLORS = {"RSI": "tab:purple", "MACD": "tab:blue", "Signal": "tab:orange",
                "RPO": "tab:green"}


def _chart_path(directory: Path, mid: str) -> Path:
    ensure_dirs()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return directory / f"{mid}_{stamp}.png"


def _metrics_box(ax, lines: list[str], loc: str = "upper left") -> None:
    x, ha = (0.02, "left") if loc == "upper left" else (0.98, "right")
    ax.text(
        x, 0.97, "\n".join(lines),
        transform=ax.transAxes, ha=ha, va="top", fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )


def _make_layout(n_main: int, selected: list[str], width: float = 13.0):
    """Figure with n_main tall panels plus one short panel per selected
    non-price indicator, all sharing the x-axis. Returns (fig, main_axes,
    panel_axes) with panel_axes keyed by indicator name."""
    panels = [i for i in PANEL_INDICATORS if i in selected]
    ratios = [3] * n_main + [1] * len(panels)
    fig, axes = plt.subplots(
        len(ratios), 1, figsize=(width, 3.0 * n_main + 1.6 * len(panels)),
        sharex=True, gridspec_kw={"height_ratios": ratios},
    )
    axes = np.atleast_1d(axes)
    return fig, axes[:n_main], dict(zip(panels, axes[n_main:]))


def _draw_indicators(price_ax, panel_axes: dict, close: pd.Series,
                     selected: list[str], start, end) -> None:
    """Draw SMA overlays on price_ax and one panel per other indicator.
    Series are computed from the full close history (no warmup gaps) and
    sliced to [start, end] for display."""
    series = display_series(close, selected)

    for name, line in series.pop("MA", {}).items():
        seg = line.loc[start:end]
        price_ax.plot(seg.index, seg.values, lw=1.0, ls="--", alpha=0.9,
                      label=name)

    for ind, lines in series.items():
        ax = panel_axes[ind]
        for name, line in lines.items():
            seg = line.loc[start:end]
            ax.plot(seg.index, seg.values, lw=1.0,
                    color=PANEL_COLORS.get(name), label=name)
        if ind == "RSI":
            ax.axhline(70, color="gray", ls=":", lw=0.8)
            ax.axhline(30, color="gray", ls=":", lw=0.8)
            ax.set_ylim(0, 100)
        else:
            ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set_ylabel(ind)
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(alpha=0.3)


def save_training_chart(mid: str, plot_data: dict,
                        selected_indicators: list[str]) -> Path:
    """Real close over the testing period with predicted 240-step trajectories
    launched from non-overlapping anchor dates, plus the metric values."""
    history: pd.DataFrame = plot_data["history"]
    X_test: pd.DataFrame = plot_data["X_test"]
    Y_pred: np.ndarray = plot_data["Y_pred"]
    metrics: dict = plot_data["metrics"]

    test_close = history["Close"].loc[X_test.index[0]:]

    fig, (ax,), panel_axes = _make_layout(1, selected_indicators)
    ax.plot(test_close.index, test_close.values, label="Real", color="black", lw=1.2)

    anchors = range(0, len(X_test), HORIZON)
    for i, a in enumerate(anchors):
        anchor_date = X_test.index[a]
        base_price = history["Close"].loc[anchor_date]
        pred_prices = base_price * (1.0 + Y_pred[a])
        pos = history.index.get_loc(anchor_date)
        future_dates = history.index[pos + 1: pos + 1 + len(pred_prices)]
        ax.plot(
            future_dates, pred_prices[: len(future_dates)],
            color="tab:red", lw=1.2, alpha=0.9,
            label="Prediction (240-step)" if i == 0 else None,
        )

    _draw_indicators(ax, panel_axes, history["Close"], selected_indicators,
                     test_close.index[0], test_close.index[-1])

    _metrics_box(ax, [
        f"MAE:  {metrics['MAE']:.4f}",
        f"MAPE: {metrics['MAPE']:.2%}",
        f"MDA:  {metrics['MDA']:.2%}",
    ])
    ax.set_title(f"{mid} — Real vs Prediction (Testing Period)")
    ax.set_ylabel("Close Price")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    path = _chart_path(PLOTS_TRAINING_DIR, mid)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_backtest_chart(mid: str, result: dict, close: pd.Series,
                        selected_indicators: list[str]) -> Path:
    """Per-window real vs predicted price, portfolio growth with loss areas
    shaded red, and one panel per selected indicator."""
    windows: pd.DataFrame = result["windows"]      # index: window end date
    portfolio: pd.Series = result["portfolio"]     # daily portfolio value
    initial_capital: float = result["initial_capital"]

    fig, (ax1, ax2), panel_axes = _make_layout(2, selected_indicators)

    ax1.plot(windows.index, windows["actual"], label="Real", color="black",
             lw=1.2, marker="o", ms=3)
    ax1.plot(windows.index, windows["predicted"], label="Predicted", color="tab:red",
             lw=1.2, marker="o", ms=3, alpha=0.85)
    _metrics_box(ax1, [
        f"MAPE: {result['window_mape']:.2%}",
        f"Total Windows: {len(windows)}",
    ])
    ax1.set_title(f"{mid} — Backtest: Real vs Predicted Price per Window")
    ax1.set_ylabel("Close Price")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    ax2.plot(portfolio.index, portfolio.values, color="tab:blue", lw=1.3,
             label="Portfolio Value")
    ax2.axhline(initial_capital, color="gray", ls="--", lw=0.8)
    ax2.fill_between(
        portfolio.index, portfolio.values, initial_capital,
        where=portfolio.values < initial_capital,
        color="red", alpha=0.25, interpolate=True, label="Loss Area",
    )
    _metrics_box(ax2, [
        f"Return: {result['return_pct']:+.2%}",
        f"Win Rate: {result['win_rate']:.2%}",
        f"Total Trades: {result['total_trades']}",
        f"Max Drawdown: {result['max_drawdown']:.2%}",
    ])
    ax2.set_title("Portfolio Growth")
    ax2.set_ylabel("Value")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)

    _draw_indicators(ax1, panel_axes, close, selected_indicators,
                     portfolio.index[0], portfolio.index[-1])

    path = _chart_path(PLOTS_BACKTEST_DIR, mid)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_forecast_chart(mid: str, history_close: pd.Series, forecast: pd.Series,
                        selected_indicators: list[str]) -> Path:
    """Last ~5 months of real data seamlessly connected to the 240-step
    forecast; indicator lines cover only the historical segment."""
    recent = history_close.iloc[-105:]  # ~5 months of trading days

    # Prepend the last real point so the forecast curve connects seamlessly.
    bridge = pd.concat([recent.iloc[[-1]], forecast])

    fig, (ax,), panel_axes = _make_layout(1, selected_indicators)
    ax.plot(recent.index, recent.values, label="Historical (5 months)",
            color="black", lw=1.3)
    ax.plot(bridge.index, bridge.values, label=f"Forecast ({HORIZON} steps)",
            color="tab:red", lw=1.3)
    ax.axvline(recent.index[-1], color="gray", ls="--", lw=0.8)

    _draw_indicators(ax, panel_axes, history_close, selected_indicators,
                     recent.index[0], recent.index[-1])

    ax.set_title(f"{mid} — {HORIZON}-Step Forecast (~1 Trading Year)")
    ax.set_ylabel("Close Price")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    path = _chart_path(PLOTS_FORECAST_DIR, mid)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
