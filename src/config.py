"""Central configuration: paths, model defaults, and .env loading."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root (this file lives in src/).
BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
MODELS_INFO_PATH = MODELS_DIR / "models_info.json"
PLOTS_TRAINING_DIR = BASE_DIR / "plots" / "training"
PLOTS_BACKTEST_DIR = BASE_DIR / "plots" / "backtest"
PLOTS_FORECAST_DIR = BASE_DIR / "plots" / "forecast"

# Forecast horizon: 240 trading days (~1 trading year), direct multi-output.
HORIZON = 240

# Data window and chronological split.
DATA_YEARS = 10
TRAIN_RATIO = 0.7

# Optional technical indicators selectable at training time.
INDICATORS = ["MA", "RSI", "MACD", "RPO"]

DEFAULT_XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
}

# Backtest defaults.
DEFAULT_TIME_WINDOW = 21          # trading days (~1 month)
DEFAULT_INITIAL_CAPITAL = 1000.0
DEFAULT_FEE_RATE = 0.0015         # 0.15% per trade side
BUY_THRESHOLD = 0.01              # predicted rise > 1% triggers a BUY
MAX_BACKTEST_YEARS = 10           # periods beyond the ~3y test set overlap training data


def model_id(symbol: str, indicators: list[str]) -> str:
    """Registry key / file stem for a symbol + indicator combination,
    e.g. ('TSLA', ['RPO', 'MA']) -> 'TSLA_MA-RPO'; no indicators -> 'TSLA'."""
    norm = [i for i in INDICATORS if i in indicators]
    return f"{symbol}_{'-'.join(norm)}" if norm else symbol


def get_target_stocks() -> list[str]:
    """Read TARGET_STOCKS from .env as a list of symbols."""
    load_dotenv(BASE_DIR / ".env")
    raw = os.getenv("TARGET_STOCKS", "")
    return [s.strip() for s in raw.split(",") if s.strip()]


def ensure_dirs() -> None:
    for d in (MODELS_DIR, PLOTS_TRAINING_DIR, PLOTS_BACKTEST_DIR, PLOTS_FORECAST_DIR):
        d.mkdir(parents=True, exist_ok=True)
