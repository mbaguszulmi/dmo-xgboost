"""Model training, persistence, and the models_info.json registry."""

import json
from datetime import datetime

import joblib
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from .config import (
    DEFAULT_XGB_PARAMS,
    MODELS_DIR,
    MODELS_INFO_PATH,
    ensure_dirs,
    model_id,
)
from .data import chronological_split, prepare_dataset
from .metrics import compute_metrics


def build_model(params: dict) -> MultiOutputRegressor:
    base = XGBRegressor(
        **params,
        tree_method="hist",
        n_jobs=1,           # parallelism is across the 240 output models instead
        random_state=42,
    )
    return MultiOutputRegressor(base, n_jobs=-1)


def train_symbol(symbol: str, selected_indicators: list[str], params: dict | None = None):
    """Train one direct multi-output model for a symbol.

    Returns (model, info_entry, plot_data) where plot_data carries everything
    the training chart needs.
    """
    params = params or dict(DEFAULT_XGB_PARAMS)
    history, X, Y = prepare_dataset(symbol, selected_indicators)
    X_train, Y_train, X_test, Y_test = chronological_split(X, Y)

    model = build_model(params)
    model.fit(X_train.values, Y_train.values)

    Y_pred = model.predict(X_test.values)
    base_close = history["Close"].loc[X_test.index].values
    metrics = compute_metrics(base_close, Y_test.values, Y_pred)

    ensure_dirs()
    mid = model_id(symbol, selected_indicators)
    model_path = MODELS_DIR / f"{mid}_model.joblib"
    joblib.dump(model, model_path)

    info_entry = {
        "symbol": symbol,
        "model_path": str(model_path.relative_to(MODELS_DIR.parent)),
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "selected_features": selected_indicators,
        "feature_columns": list(X.columns),
        "parameters": params,
        "metrics": metrics,
    }
    save_model_info(mid, info_entry)

    plot_data = {
        "history": history,
        "X_test": X_test,
        "Y_test": Y_test,
        "Y_pred": Y_pred,
        "metrics": metrics,
    }
    return model, info_entry, plot_data


def load_models_info() -> dict:
    if MODELS_INFO_PATH.exists():
        with open(MODELS_INFO_PATH) as f:
            return json.load(f)
    return {}


def save_model_info(mid: str, entry: dict) -> None:
    info = load_models_info()
    info[mid] = entry
    with open(MODELS_INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)


def load_model(mid: str) -> MultiOutputRegressor:
    """Load a model by its registry key (symbol + indicator variant)."""
    info = load_models_info()
    if mid not in info:
        raise KeyError(f"No trained model registered for '{mid}'")
    return joblib.load(MODELS_DIR.parent / info[mid]["model_path"])


def predict_returns(model: MultiOutputRegressor, X: np.ndarray) -> np.ndarray:
    """Predict the (n, HORIZON) return matrix for feature rows X."""
    return model.predict(np.atleast_2d(X))
