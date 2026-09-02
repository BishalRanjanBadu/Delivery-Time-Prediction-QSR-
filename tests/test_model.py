"""
Phase-2 test suite (CI hard-gate before any Docker build).

Coverage required by the v3 instruction:
  * technical — model trains, predicts a number, in a sane range
  * business-invariant — domain truths the model must respect (Jam traffic slower
    than Low; longer distance → longer time). Analog of the reference's
    "rideshare predicts higher than personal".
  * leakage regression — statistics must be fit on train rows only
  * call-order guard — split() must run before fit() at the real call site
  * MAPE quality gate — MAPE < gate

The leakage + call-order tests are the reason a pre-split-preprocessing
regression (the reference's bug) cannot reach production silently.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import DeliveryPreprocessor, split_raw, TARGET  # noqa: E402
import train as T  # noqa: E402
import predict as P  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "zomato_raw.csv")


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def splits(raw):
    return split_raw(raw)


@pytest.fixture(scope="module")
def trained(raw):
    return T.train(raw_df=raw, n_iter=6, persist=False)


# ---------------- technical ----------------
def test_transform_no_nulls(splits):
    train_df, test_df = splits
    pre = DeliveryPreprocessor()
    Xtr = pre.fit_transform(train_df)
    Xte = pre.transform(test_df)
    assert Xtr[pre.feature_names_].isna().sum().sum() == 0
    assert Xte[pre.feature_names_].isna().sum().sum() == 0


def test_distance_km_kept_raw_coords_dropped(splits):
    train_df, _ = splits
    pre = DeliveryPreprocessor().fit(train_df)
    assert "distance_km" in pre.feature_names_
    for c in ["Restaurant_latitude", "Delivery_location_latitude",
              "Restaurant_longitude", "Delivery_location_longitude"]:
        assert c not in pre.feature_names_


def test_prediction_is_number_in_range(raw, trained):
    pre, model = trained["preprocessor"], trained["model"]
    preds = P.predict_df(raw.drop(columns=[TARGET]).head(20), preprocessor=pre, model=model)
    assert len(preds) == 20
    assert np.isfinite(preds).all()
    assert preds.min() > 0 and preds.max() < 120


def test_metrics_not_suspiciously_perfect(trained):
    m = trained["metrics"]
    assert m["mae"] > 0.5           # delivery time has irreducible noise
    assert m["r2"] < 0.99           # ~1.0 would signal a leak


# ---------------- business-invariant ----------------
def _base_record(raw):
    r = raw.drop(columns=[TARGET]).iloc[0].to_dict()
    return r


def test_jam_traffic_slower_than_low(raw, trained):
    """Holding everything else fixed, Jam traffic must predict >= Low traffic."""
    pre, model = trained["preprocessor"], trained["model"]
    base = _base_record(raw)
    low = dict(base); low["Road_traffic_density"] = "Low"
    jam = dict(base); jam["Road_traffic_density"] = "Jam"
    df = pd.DataFrame([low, jam])
    preds = P.predict_df(df, preprocessor=pre, model=model)
    assert preds.iloc[1] >= preds.iloc[0], "Jam should not be faster than Low traffic"


def test_longer_distance_not_faster(raw, trained):
    pre, model = trained["preprocessor"], trained["model"]
    base = _base_record(raw)
    near = dict(base); near["distance_km"] = 2.0
    far = dict(base); far["distance_km"] = 18.0
    df = pd.DataFrame([near, far])
    preds = P.predict_df(df, preprocessor=pre, model=model)
    assert preds.iloc[1] >= preds.iloc[0], "Longer distance should not predict shorter time"


def test_eta_band_and_window_valid(raw, trained):
    pre, model, _ = (trained["preprocessor"], trained["model"], None)
    # exercise the business layer directly via injected artifacts
    df = raw.drop(columns=[TARGET]).head(3)
    enc = pre.transform(df)
    X = enc[pre.feature_names_]
    mins = [float(v) for v in model.predict(X)]
    for m in mins:
        band = P._eta_band(m)
        assert band in {"Fast", "Average", "Slow"}


# ---------------- leakage regression ----------------
def test_leakage_stats_from_train_only(raw, splits):
    train_df, _ = splits
    pre = DeliveryPreprocessor().fit(train_df)
    eng_train = DeliveryPreprocessor._engineer(train_df)
    assert pre.impute_stats_["Delivery_person_Age"] == pytest.approx(
        float(eng_train["Delivery_person_Age"].median())
    )


def test_presplit_pattern_is_detectably_different(raw, splits):
    """Fitting on the full frame (the reference's bug) must yield different scaler
    means than fitting on train only. If identical, leakage would be invisible."""
    train_df, _ = splits
    right = DeliveryPreprocessor().fit(train_df)
    wrong = DeliveryPreprocessor().fit(raw)
    assert not np.allclose(right.scaler_.mean_, wrong.scaler_.mean_)


def test_transform_does_not_refit(splits):
    train_df, test_df = splits
    pre = DeliveryPreprocessor().fit(train_df)
    before = dict(pre.impute_stats_), dict(pre.outlier_bounds_), list(pre.feature_names_)
    pre.transform(test_df)
    after = dict(pre.impute_stats_), dict(pre.outlier_bounds_), list(pre.feature_names_)
    assert before == after


# ---------------- call-order guard ----------------
def test_train_splits_before_fit(monkeypatch, raw):
    calls = []
    real_split = T.split_raw
    real_ft = DeliveryPreprocessor.fit_transform

    def spy_split(df, **kw):
        calls.append("split"); return real_split(df, **kw)

    def spy_ft(self, df):
        calls.append("fit"); return real_ft(self, df)

    monkeypatch.setattr(T, "split_raw", spy_split)
    monkeypatch.setattr(DeliveryPreprocessor, "fit_transform", spy_ft)
    T.train(raw_df=raw, n_iter=2, persist=False)
    assert calls.index("split") < calls.index("fit")


# ---------------- MAPE quality gate ----------------
def test_mape_gate(trained):
    m = trained["metrics"]
    assert m["mape"] < T.MAPE_GATE
    assert m["gate_passed"] is True


# ---------------- API end-to-end (mocked S3) ----------------
def test_api_end_to_end(raw):
    moto = pytest.importorskip("moto")
    import boto3
    os.environ["S3_BUCKET"] = "test-bucket"
    os.environ["AWS_REGION"] = "ap-south-1"
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
    os.environ["MODEL_VERSION"] = "v-test"
    with moto.mock_aws():
        boto3.client("s3", region_name="ap-south-1").create_bucket(
            Bucket="test-bucket",
            CreateBucketConfiguration={"LocationConstraint": "ap-south-1"},
        )
        import train as T2
        import predict as P2
        T2.train(raw_df=raw, n_iter=2, persist=True)
        P2.reset_cache()
        from fastapi.testclient import TestClient
        import api
        client = TestClient(api.app)
        assert client.get("/health").json()["status"] == "ok"
        assert "mape" in client.get("/model_info").json()
        recs = raw.drop(columns=[TARGET]).head(3).to_dict("records")
        r = client.post("/predict", json={"records": recs})
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) == 3
        assert all(res["eta_band"] in {"Fast", "Average", "Slow"} for res in results)
        assert all(0 < res["predicted_time_min"] < 120 for res in results)
