"""
Inference for the QSR delivery-time model (Phase 2).

TRANSFORM-ONLY: loads the fitted preprocessor + model from S3 and applies the
statistics learned on the training split. Nothing is fit here, so inference
cannot introduce leakage or train/serve skew.

Business layer (derived FROM the model output, never fed back as a feature):
  * eta_band          — Fast / Average / Slow bucket, for ops routing
  * delivery_window   — predicted +/- the model's held-out MAE
  * shap_explanation  — top per-prediction contributions (ENABLE_SHAP=0 to skip)

FALLBACK CONTRACT (was missing): if the model or preprocessor cannot produce a
prediction, `predict_records` degrades to a documented rule-based estimate and
marks the record `source="fallback"` instead of raising an undefined 5xx.
Order: model -> rule-based default -> graceful error. The caller can key alerts
off `source`.
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache

import pandas as pd

import s3_io

logger = logging.getLogger(__name__)

PREPROCESSOR_KEY = "models/preprocessor.pkl"
MODEL_KEY = "models/best_model.pkl"
METRICS_KEY = "models/model_metrics.json"

ETA_BANDS = [(0, 20, "Fast"), (20, 34, "Average"), (34, float("inf"), "Slow")]

# Rule-based fallback, calibrated from the Phase-1 training distribution.
# Deliberately crude: it exists so a model outage degrades service instead of
# failing it. Any use of it should page.
FALLBACK_BASE_MIN = float(os.environ.get("FALLBACK_BASE_MIN", "20"))
FALLBACK_MIN_PER_KM = float(os.environ.get("FALLBACK_MIN_PER_KM", "1.6"))
FALLBACK_TRAFFIC_ADD = {"Low": 0.0, "Medium": 3.0, "High": 6.0, "Jam": 10.0}


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


def _rule_based_minutes(rec: dict) -> float:
    """Documented degraded-mode estimate. Used only when the model path fails."""
    dist = rec.get("distance_km")
    try:
        dist = float(dist)
    except (TypeError, ValueError):
        dist = 5.0
    traffic = FALLBACK_TRAFFIC_ADD.get(str(rec.get("Road_traffic_density", "")).strip(), 4.0)
    festival = 8.0 if str(rec.get("Festival", "")).strip() == "Yes" else 0.0
    return FALLBACK_BASE_MIN + FALLBACK_MIN_PER_KM * dist + traffic + festival


def _package(minutes: float, mae: float, source: str) -> dict:
    minutes = float(minutes)
    return {
        "predicted_time_min": round(minutes, 1),
        "eta_band": _eta_band(minutes),
        "delivery_window_min": [round(max(0.0, minutes - mae), 1), round(minutes + mae, 1)],
        "source": source,
    }


def predict_records(records: list[dict], with_explanation: bool | None = None) -> list[dict]:
    """Predict for raw JSON records, returning the full business layer per record.

    Never raises for a model-side failure: falls back to the rule-based estimate
    and tags the record so callers/alerts can see degraded mode.
    """
    if with_explanation is None:
        # ENABLE_SHAP=0 lets memory-tight nodes (e.g. free-tier t3.small) skip SHAP
        with_explanation = os.environ.get("ENABLE_SHAP", "1") != "0"

    try:
        pre, model, metrics = _load_artifacts()
        df = pd.DataFrame(records)
        enc = pre.transform(df)
        X = enc[pre.feature_names_]
        preds = model.predict(X)
        # MAE defaults to a non-zero width so the window is never a false point.
        mae = float(metrics.get("mae") or 4.0)
    except Exception as e:  # noqa: BLE001
        logger.error("MODEL PATH FAILED (%s) -> rule-based fallback", e)
        return [_package(_rule_based_minutes(r), 6.0, "fallback") for r in records]

    out = []
    for i, minutes in enumerate(preds):
        rec = _package(minutes, mae, "model")
        if with_explanation:
            rec["shap_explanation"] = _shap_top(model, X.iloc[[i]])
        out.append(rec)
    return out
