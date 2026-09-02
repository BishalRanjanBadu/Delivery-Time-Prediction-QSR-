"""
Training entrypoint for the Zomato delivery-time model (Phase 2 — deployment).

Execution order enforces the leakage boundary:
    raw = read_csv_s3('raw/...')
    train_df, test_df = split_raw(raw)      # split FIRST
    pre.fit_transform(train_df)             # learn on train only
    pre.transform(test_df)                  # apply to test
Nothing is fit on a frame containing test rows.

Promotion (matches the reference's lighter approach — MLflow + model_metrics.json,
no formal Model Registry): after evaluation the new model is compared against the
current models/model_metrics.json in S3. It is promoted (artifacts overwritten in
S3) only if it (a) passes the MAPE quality gate AND (b) is at least as good as the
incumbent on the gate metric. A failing or worse retrain leaves production
untouched.

Experiment tracking (MLflow) is a Phase-3 concern and is intentionally NOT wired
here — Phase 2 promotes via model_metrics.json comparison, which is all the deploy
decision needs.
"""
from __future__ import annotations

import os
import json
import logging

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
import xgboost as xgb

from preprocess import DeliveryPreprocessor, split_raw, TARGET
import s3_io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Business quality gate. Phase-1 verified test MAPE ≈ 13.3%; 18% leaves headroom
# while still catching a real regression. CI reads the same value from
# MODEL_MAPE_GATE so the training gate and the deploy gate never diverge.
MAPE_GATE = float(os.environ.get("MODEL_MAPE_GATE", "0.18"))

RAW_KEY = os.environ.get("RAW_KEY", "raw/zomato_raw.csv")
PREPROCESSOR_KEY = "models/preprocessor.pkl"
MODEL_KEY = "models/best_model.pkl"
FEATURES_KEY = "models/feature_names.pkl"
METRICS_KEY = "models/model_metrics.json"

PARAM_DIST = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, pred)),
        "mape": float(mean_absolute_percentage_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
    }


def _load_incumbent_metrics() -> dict | None:
    try:
        raw = s3_io.get_client().get_object(
            Bucket=s3_io._bucket(None), Key=METRICS_KEY
        )["Body"].read()
        return json.loads(raw)
    except Exception:
        return None


def train(raw_df=None, n_iter: int = 20, persist: bool = True) -> dict:
    if raw_df is None:
        raw_df = s3_io.read_csv_s3(RAW_KEY)

    # --- split BEFORE any fit ---
    train_df, test_df = split_raw(raw_df)

    pre = DeliveryPreprocessor()
    train_enc = pre.fit_transform(train_df)     # learns on train only
    test_enc = pre.transform(test_df)           # applies to test

    feats = pre.feature_names_
    X_train, y_train = train_enc[feats], train_enc[TARGET]
    X_test, y_test = test_enc[feats], test_enc[TARGET]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=-1),
        PARAM_DIST,
        n_iter=n_iter,
        cv=kf,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)                 # CV folds internal to train
    model = search.best_estimator_

    from datetime import datetime, timezone
    metrics = evaluate(model, X_test, y_test)
    metrics["cv_mae"] = float(-search.best_score_)
    metrics["best_params"] = search.best_params_
    metrics["training_date"] = datetime.now(timezone.utc).isoformat()
    metrics["mape_gate"] = MAPE_GATE
    logger.info("Metrics: %s", {k: metrics[k] for k in ("mae", "mape", "rmse", "r2")})

    # --- gate + compare-with-incumbent promotion ---
    gate_passed = metrics["mape"] < MAPE_GATE
    metrics["gate_passed"] = gate_passed

    incumbent = _load_incumbent_metrics()
    better_than_incumbent = (incumbent is None) or (metrics["mape"] <= incumbent.get("mape", float("inf")))
    metrics["incumbent_mape"] = None if incumbent is None else incumbent.get("mape")
    promote = gate_passed and better_than_incumbent
    metrics["promoted"] = promote

    if not gate_passed:
        logger.error("QUALITY GATE FAILED: MAPE %.4f >= %.4f — not promoted.",
                     metrics["mape"], MAPE_GATE)
    elif not better_than_incumbent:
        logger.warning("Gate passed but worse than incumbent (%.4f vs %.4f) — not promoted.",
                       metrics["mape"], incumbent.get("mape"))

    if persist and promote:
        s3_io.save_pickle_s3(pre, PREPROCESSOR_KEY)
        s3_io.save_pickle_s3(model, MODEL_KEY)
        s3_io.save_pickle_s3(feats, FEATURES_KEY)
        s3_io.save_json_s3(metrics, METRICS_KEY)
        logger.info("PROMOTED — artifacts written to S3.")

    return {"preprocessor": pre, "model": model, "features": feats, "metrics": metrics}


if __name__ == "__main__":
    out = train()
    m = out["metrics"]
    print(f"MAPE={m['mape']*100:.2f}%  MAE={m['mae']:.3f}  R2={m['r2']:.3f}  "
          f"gate_passed={m['gate_passed']}  promoted={m['promoted']}")
    raise SystemExit(0 if m["gate_passed"] else 1)
