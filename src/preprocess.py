"""
Preprocessing for the Zomato delivery-time model.

LEAKAGE BOUNDARY — the whole point of this module:
  * `DeliveryPreprocessor.fit()` learns every cross-row statistic
    (imputation medians, outlier bounds, one-hot categories, VIF-based drops,
    scaler) and is called EXACTLY ONCE, on the TRAIN split only.
  * `.transform()` applies already-learned statistics and is called for test,
    inference, and every retrain cycle. It fits nothing.

The fit/transform split is not enough on its own — the CALL SITE has to split
before fitting. `train.py` and `pipeline/retrain.py` both do:
      train_df, test_df = split(raw)
      pre.fit_transform(train_df)     # learns on train
      pre.transform(test_df)          # applies to test
Never `pre.fit(full_frame)`. The leakage regression test in tests/ asserts this
by fitting on train and checking that test statistics never leak in.

Deterministic per-row feature engineering (date/time decomposition, distance is
already present) has no cross-row statistic and is applied identically in both
fit and transform — it cannot leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

TARGET = "Time_taken (min)"

# Columns dropped as leakage / non-generalizable identifiers.
LEAKAGE_COLS = ["delivery_speed"]          # direct discretization of the target
ID_COLS = ["ID", "Delivery_person_ID"]     # identifiers, not features

# Fixed domain maps — these are knowledge, not statistics learned from data,
# so applying them to any row (train or test) is not leakage.
ORDINAL_MAP = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
BINARY_MAP = {"No": 0, "Yes": 1}

IMPUTE_NUM = ["Delivery_person_Age", "Delivery_person_Ratings", "order_hour", "prep_time_min"]
IMPUTE_MODE = ["multiple_deliveries"]
OUTLIER_COLS = ["Delivery_person_Age", "Delivery_person_Ratings", "distance_km", "prep_time_min"]
OHE_COLS = ["Weather_conditions", "Type_of_order", "Type_of_vehicle", "City"]

# Raw coordinate columns dropped in favour of the deterministic, interpretable
# distance_km (which is collinear with them). Judgment call carried over from
# notebook 06: keep distance_km, drop the redundant raw coordinates.
RAW_COORD_COLS = [
    "Restaurant_latitude", "Restaurant_longitude",
    "Delivery_location_latitude", "Delivery_location_longitude",
]

VIF_THRESHOLD = 10.0


class DeliveryPreprocessor:
    """Stateful preprocessor. Fit on train only; transform everywhere."""

    def __init__(self, vif_threshold: float = VIF_THRESHOLD):
        self.vif_threshold = vif_threshold
        self.fitted_ = False
        # learned state (populated by fit)
        self.impute_stats_: dict = {}
        self.outlier_bounds_: dict = {}
        self.ohe_: OneHotEncoder | None = None
        self.scaler_: StandardScaler | None = None
        self.vif_dropped_: list[str] = []
        self.feature_names_: list[str] = []

    # ---- deterministic, per-row: safe in both fit and transform ----
    @staticmethod
    def _engineer(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop(columns=LEAKAGE_COLS, errors="ignore")
        df = df.drop(columns=ID_COLS, errors="ignore")

        order_date = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")
        t_ord = pd.to_datetime(df["Time_Orderd"], format="%H:%M", errors="coerce")
        t_pick = pd.to_datetime(df["Time_Order_picked"], format="%H:%M", errors="coerce")

        df["order_hour"] = t_ord.dt.hour
        df["day_of_week"] = order_date.dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        prep = (t_pick - t_ord).dt.total_seconds() / 60.0
        df["prep_time_min"] = np.where(prep < 0, prep + 24 * 60, prep)  # midnight rollover

        df = df.drop(
            columns=["Order_Date", "Time_Orderd", "Time_Order_picked"], errors="ignore"
        )
        return df

    def _apply_fixed_maps(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Road_traffic_density"] = df["Road_traffic_density"].map(ORDINAL_MAP)
        df["Festival"] = df["Festival"].map(BINARY_MAP)
        return df

    # ---- FIT: learns statistics on TRAIN ONLY ----
    def fit(self, train_df: pd.DataFrame) -> "DeliveryPreprocessor":
        df = self._engineer(train_df)

        # imputation stats — from train
        for c in IMPUTE_NUM:
            self.impute_stats_[c] = float(df[c].median())
        for c in IMPUTE_MODE:
            self.impute_stats_[c] = df[c].mode().iloc[0]
        for c, v in self.impute_stats_.items():
            df[c] = df[c].fillna(v)

        # outlier bounds — from train (3x IQR)
        for c in OUTLIER_COLS:
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.outlier_bounds_[c] = (q1 - 3 * iqr, q3 + 3 * iqr)
            lo, hi = self.outlier_bounds_[c]
            df[c] = df[c].clip(lo, hi)

        df = self._apply_fixed_maps(df)

        # one-hot encoder — fit on train
        self.ohe_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")
        self.ohe_.fit(df[OHE_COLS])
        df = self._concat_ohe(df)

        # candidate feature columns (drop redundant raw coords up front)
        df = df.drop(columns=RAW_COORD_COLS, errors="ignore")
        feat_cols = [c for c in df.columns if c != TARGET]

        # VIF — computed on train. Keep distance_km even if flagged; drop only
        # other >threshold offenders. (Here the raw coords are already gone, so
        # distance_km stops being collinear; this guards any future additions.)
        self.vif_dropped_ = self._vif_drops(df[feat_cols])
        feat_cols = [c for c in feat_cols if c not in self.vif_dropped_]

        # scaler — fit on train
        self.scaler_ = StandardScaler()
        self.scaler_.fit(df[feat_cols])

        self.feature_names_ = feat_cols
        self.fitted_ = True
        return self

    def _vif_drops(self, X: pd.DataFrame) -> list[str]:
        # standardize for numerical stability, then compute VIF per column
        tmp_scaler = StandardScaler()
        Xs = tmp_scaler.fit_transform(X)
        drops = []
        for i, col in enumerate(X.columns):
            try:
                v = variance_inflation_factor(Xs, i)
            except Exception:
                v = np.nan
            if col == "distance_km":
                continue  # keep the interpretable feature regardless of VIF
            if np.isfinite(v) and v > self.vif_threshold:
                drops.append(col)
        return drops

    def _concat_ohe(self, df: pd.DataFrame) -> pd.DataFrame:
        arr = self.ohe_.transform(df[OHE_COLS])
        ohe_df = pd.DataFrame(
            arr, columns=self.ohe_.get_feature_names_out(OHE_COLS), index=df.index
        )
        return pd.concat([df.drop(columns=OHE_COLS), ohe_df], axis=1)

    # ---- TRANSFORM: applies learned statistics. Fits nothing. ----
    def transform(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("transform() called before fit().")
        df = self._engineer(df_in)

        for c, v in self.impute_stats_.items():
            if c in df.columns:
                df[c] = df[c].fillna(v)
        for c, (lo, hi) in self.outlier_bounds_.items():
            if c in df.columns:
                df[c] = df[c].clip(lo, hi)

        df = self._apply_fixed_maps(df)
        df = self._concat_ohe(df)
        df = df.drop(columns=RAW_COORD_COLS, errors="ignore")

        # align to learned feature set (add any missing OHE cols as 0, order fixed)
        for c in self.feature_names_:
            if c not in df.columns:
                df[c] = 0.0
        X = df[self.feature_names_].copy()
        X[self.feature_names_] = self.scaler_.transform(X[self.feature_names_])

        if TARGET in df_in.columns:
            X[TARGET] = df_in[TARGET].values
        return X

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)


def split_raw(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Single source of the train/test split. Called BEFORE any fit()."""
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
