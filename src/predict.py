"""
Inference for the Zomato delivery-time model (Phase 2).

TRANSFORM-ONLY: loads the fitted preprocessor + model from S3 and applies the
statistics learned on the training split. Nothing is fit here, so inference
cannot introduce leakage or train/serve skew — it is the same `.transform()` path
used for test and (in Phase 3) retraining.

Business layer (post-prediction mapping — derived from the model output, NOT fed
back as a feature, so no leakage):
  * eta_band   — Fast / Average / Slow bucket of the predicted minutes, for ops
                 routing and customer messaging.
  * delivery_window — predicted ± the model's held-out MAE, a calibrated range
                 rather than a false point-precision.
  * shap_explanation — top per-prediction feature contributions.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache

import numpy as np
import pandas as pd

import s3_io
from preprocess import TARGET

logger = logging.getLogger(__name__)

PREPROCESSOR_KEY = "models/preprocessor.pkl"
MODEL_KEY = "models/best_model.pkl"
METRICS_KEY = "models/model_metrics.json"

# ETA bands (minutes). Post-prediction buckets for the business layer.
ETA_BANDS = [(0, 20, "Fast"), (20, 34, "Average"), (34, float("inf"), "Slow")]


@lru_cache(maxsize=1)
def _load_artifacts():
    pre = s3_io.load_pickle_s3(PREPROCESSOR_KEY)
    model = s3_io.load_pickle_s3(MODEL_KEY)
    try:
        metrics = json.loads(
            s3_io.get_client().get_object(
                Bucket=s3_io._bucket(None), Key=METRICS_KEY
            )["Body"].read()
        )
    except Exception:
        metrics = {}
    logger.info("Loaded preprocessor + model + metrics from S3.")
    return pre, model, metrics


def reset_cache() -> None:
    _load_artifacts.cache_clear()


def get_model_info() -> dict:
    _, _, metrics = _load_artifacts()
    return {
        "mape": metrics.get("mape"),
        "mae": metrics.get("mae"),
        "r2": metrics.get("r2"),
        "training_date": metrics.get("training_date"),
    }


def _eta_band(minutes: float) -> str:
    for lo, hi, label in ETA_BANDS:
        if lo <= minutes < hi:
            return label
    return "Slow"


def predict_df(raw_df: pd.DataFrame, preprocessor=None, model=None) -> pd.Series:
    if preprocessor is None or model is None:
        preprocessor, model, _ = _load_artifacts()
    enc = preprocessor.transform(raw_df)          # transform-only
    X = enc[preprocessor.feature_names_]
    preds = model.predict(X)
    return pd.Series(preds, index=raw_df.index, name="predicted_time_min")


def _shap_top(model, X_row: pd.DataFrame, k: int = 5) -> dict:
    try:
        import shap
        sv = shap.TreeExplainer(model).shap_values(X_row)
        contrib = pd.Series(sv[0], index=X_row.columns)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(k)
        return {feat: round(float(val), 3) for feat, val in top.items()}
    except Exception as e:  # noqa: BLE001
        logger.warning("SHAP unavailable (%s)", e)
        return {}


def predict_records(records: list[dict], with_explanation: bool = True) -> list[dict]:
    """Predict for raw JSON records, returning the full business layer per record."""
    pre, model, metrics = _load_artifacts()
    df = pd.DataFrame(records)
    enc = pre.transform(df)
    X = enc[pre.feature_names_]
    preds = model.predict(X)
    mae = float(metrics.get("mae", 0.0))

    out = []
    for i, minutes in enumerate(preds):
        minutes = float(minutes)
        lo = max(0.0, minutes - mae)
        hi = minutes + mae
        rec = {
            "predicted_time_min": round(minutes, 1),
            "eta_band": _eta_band(minutes),
            "delivery_window_min": [round(lo, 1), round(hi, 1)],
        }
        if with_explanation:
            rec["shap_explanation"] = _shap_top(model, X.iloc[[i]])
        out.append(rec)
    return out
