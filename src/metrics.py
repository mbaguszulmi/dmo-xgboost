"""Evaluation metrics over the test set, computed in price space."""

import numpy as np


def compute_metrics(base_close: np.ndarray, Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    """MAE / MAPE in price terms and MDA over all (row, horizon) pairs.

    base_close: (n,) close price at each anchor date t.
    Y_true / Y_pred: (n, HORIZON) returns relative to the anchor close.
    """
    base = base_close.reshape(-1, 1)
    true_prices = base * (1.0 + Y_true)
    pred_prices = base * (1.0 + Y_pred)

    mae = float(np.mean(np.abs(true_prices - pred_prices)))
    mape = float(np.mean(np.abs((true_prices - pred_prices) / true_prices)))
    # Direction of the move from the anchor price at each horizon.
    mda = float(np.mean(np.sign(Y_true) == np.sign(Y_pred)))

    return {"MAE": round(mae, 4), "MAPE": round(mape, 4), "MDA": round(mda, 4)}
