"""
Preprocessing for the QSR delivery-time model.

LEAKAGE BOUNDARY — the whole point of this module:
  * `DeliveryPreprocessor.fit()` learns every cross-row statistic
    (imputation medians/modes, outlier bounds, one-hot categories, VIF-based
    drops, scaler) and is called EXACTLY ONCE, on the TRAIN split only.
  * `.transform()` applies already-learned statistics and is called for test,
    inference, and every retrain cycle. It fits nothing.

The fit/transform split is not enough on its own — the CALL SITE has to split
before fitting. `train.py` does:
      train_df, test_df = split_raw(raw)
      pre.fit_transform(train_df)     # learns on train
      pre.transform(test_df)          # applies to test
Never `pre.fit(full_frame)`.

Deterministic per-row work (sentinel cleaning, date/time decomposition,
haversine distance, fixed ordinal/binary maps) has no cross-row statistic and is
applied identically in fit and transform — it cannot leak.

FIXES vs the previous revision
  1. `distance_km` is now DERIVED (haversine) when the raw frame does not carry
     it. Previously OUTLIER_COLS referenced a column `_engineer` never created,
     so `fit()` raised KeyError on the Kaggle-schema raw file.
  2. Sentinel strings ("NaN", "NaN ", "conditions Sunny", trailing spaces) are
     normalised to real NaN, and the fixed ordinal/binary maps now run BEFORE
     imputation. Previously mapping ran after imputation, so every unmapped
     value became a NaN that nothing filled — it flowed into StandardScaler and
     out to the model.
  3. `transform()` asserts the encoded frame is NaN-free, so this class of bug
     fails loudly at train/serve time instead of silently degrading predictions.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

TARGET = "Time_taken (min)"

# Columns dropped as leakage / non-generalizable identifiers.
LEAKAGE_COLS = ["delivery_speed"]          # direct discretization of the target
ID_COLS = ["ID", "Delivery_person_ID"]     # identifiers, not features

# Fixed domain maps — knowledge, not statistics learned from data, so applying
# them to any row (train or test) is not leakage.
ORDINAL_MAP = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
BINARY_MAP = {"No": 0, "Yes": 1}

# Values the source CSV uses to mean "missing". The Kaggle export writes the
# literal string "NaN" (often with a trailing space) rather than an empty field.
SENTINELS = {"NaN", "nan", "NAN", "", "-", "null", "None", "?"}

# Some exports prefix categorical values with the column name
# ("conditions Sunny", "(min) 24"). Stripped here so the maps/OHE see clean levels.
VALUE_PREFIXES = ("conditions ", "(min) ")

NUM_COLS = [
    "Delivery_person_Age", "Delivery_person_Ratings", "Vehicle_condition",
    "multiple_deliveries", "distance_km",
    "Restaurant_latitude", "Restaurant_longitude",
    "Delivery_location_latitude", "Delivery_location_longitude",
]

# Imputed with the TRAIN median.
IMPUTE_NUM = [
    "Delivery_person_Age", "Delivery_person_Ratings", "order_hour",
    "prep_time_min", "distance_km", "day_of_week",
]
# Imputed with the TRAIN mode (discrete / already-mapped ordinals).
IMPUTE_MODE = [
    "multiple_deliveries", "Vehicle_condition",
    "Road_traffic_density", "Festival", "is_weekend",
]
OUTLIER_COLS = ["Delivery_person_Age", "Delivery_person_Ratings", "distance_km", "prep_time_min"]
OHE_COLS = ["Weather_conditions", "Type_of_order", "Type_of_vehicle", "City"]

# Raw coordinates are dropped in favour of the deterministic, interpretable
# distance_km derived from them. Judgment call from notebook 06: keep the
# interpretable feature, drop the redundant collinear raw inputs.
RAW_COORD_COLS = [
    "Restaurant_latitude", "Restaurant_longitude",
    "Delivery_location_latitude", "Delivery_location_longitude",
]

VIF_THRESHOLD = 10.0
EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km. Deterministic per row — no leakage."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype="float64"))
                              for v in (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class DeliveryPreprocessor:
    """Stateful preprocessor. Fit on train only; transform everywhere."""

    def __init__(self, vif_threshold: float = VIF_THRESHOLD):
        self.vif_threshold = vif_threshold
        self.fitted_ = False
        self.impute_stats_: dict = {}
        self.outlier_bounds_: dict = {}
        self.ohe_: OneHotEncoder | None = None
        self.scaler_: StandardScaler | None = None
        self.vif_dropped_: list[str] = []
        self.feature_names_: list[str] = []

    # ---- deterministic, per-row: identical in fit and transform ----
    @staticmethod
    def _clean_strings(df: pd.DataFrame) -> pd.DataFrame:
        for c in df.columns:
            if df[c].dtype == object:
                s = df[c].astype("string").str.strip()
                for p in VALUE_PREFIXES:
                    s = s.str.removeprefix(p)
                s = s.str.strip()
                df[c] = s.where(~s.isin(SENTINELS), other=pd.NA)
        return df

    @staticmethod
    def _engineer(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop(columns=LEAKAGE_COLS, errors="ignore")
        df = df.drop(columns=ID_COLS, errors="ignore")

        df = DeliveryPreprocessor._clean_strings(df)

        # numeric coercion AFTER sentinel cleaning ("NaN " -> <NA> -> nan)
        for c in NUM_COLS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if TARGET in df.columns:
            df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

        order_date = pd.to_datetime(df.get("Order_Date"), format="%d-%m-%Y", errors="coerce")
        t_ord = pd.to_datetime(df.get("Time_Orderd"), format="%H:%M", errors="coerce")
        t_pick = pd.to_datetime(df.get("Time_Order_picked"), format="%H:%M", errors="coerce")

        df["order_hour"] = t_ord.dt.hour
        df["day_of_week"] = order_date.dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("float64")

        prep = (t_pick - t_ord).dt.total_seconds() / 60.0
        df["prep_time_min"] = np.where(prep < 0, prep + 24 * 60, prep)

        # --- distance_km: use the supplied column, else derive it ---
        have_coords = all(c in df.columns for c in RAW_COORD_COLS)
        if "distance_km" not in df.columns:
            df["distance_km"] = np.nan
        if have_coords:
            # Kaggle export carries sign-flipped / near-zero junk coordinates.
            coords = {c: df[c].abs() for c in RAW_COORD_COLS}
            derived = _haversine_km(
                coords["Restaurant_latitude"], coords["Restaurant_longitude"],
                coords["Delivery_location_latitude"], coords["Delivery_location_longitude"],
            )
            derived = pd.Series(derived, index=df.index)
            # junk rows (either endpoint ~0) are left as NaN for the imputer
            bad = (coords["Restaurant_latitude"] < 1) | (coords["Delivery_location_latitude"] < 1)
            derived = derived.mask(bad)
            df["distance_km"] = df["distance_km"].fillna(derived)

        df = df.drop(columns=["Order_Date", "Time_Orderd", "Time_Order_picked"], errors="ignore")
        return df

    @staticmethod
    def _apply_fixed_maps(df: pd.DataFrame) -> pd.DataFrame:
        """Ordinal/binary maps. Run BEFORE imputation so unmapped values are
        imputed rather than surviving as un-fillable NaN."""
        df = df.copy()
        if "Road_traffic_density" in df.columns:
            df["Road_traffic_density"] = (
                df["Road_traffic_density"].map(ORDINAL_MAP).astype("float64")
            )
        if "Festival" in df.columns:
            df["Festival"] = df["Festival"].map(BINARY_MAP).astype("float64")
        return df

    def _prepare(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Deterministic pipeline shared by fit and transform."""
        return self._apply_fixed_maps(self._engineer(df_in))

    # ---- FIT: learns statistics on TRAIN ONLY ----
    def fit(self, train_df: pd.DataFrame) -> "DeliveryPreprocessor":
        df = self._prepare(train_df)

        for c in IMPUTE_NUM:
            if c in df.columns:
                med = df[c].median()
                self.impute_stats_[c] = float(med) if pd.notna(med) else 0.0
        for c in IMPUTE_MODE:
            if c in df.columns:
                mode = df[c].mode(dropna=True)
                self.impute_stats_[c] = (
                    float(mode.iloc[0]) if len(mode) else float(df[c].median() or 0.0)
                )
        df = self._impute(df)

        for c in OUTLIER_COLS:
            if c not in df.columns:
                raise KeyError(
                    f"OUTLIER_COLS references '{c}' which is not present after "
                    f"_engineer(). Columns: {sorted(df.columns)}"
                )
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.outlier_bounds_[c] = (float(q1 - 3 * iqr), float(q3 + 3 * iqr))
            lo, hi = self.outlier_bounds_[c]
            df[c] = df[c].clip(lo, hi)

        self.ohe_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")
        self.ohe_.fit(self._ohe_frame(df))
        df = self._concat_ohe(df)

        df = df.drop(columns=RAW_COORD_COLS, errors="ignore")
        feat_cols = [c for c in df.columns if c != TARGET]

        self.vif_dropped_ = self._vif_drops(df[feat_cols])
        feat_cols = [c for c in feat_cols if c not in self.vif_dropped_]

        self.scaler_ = StandardScaler()
        self.scaler_.fit(df[feat_cols])

        self.feature_names_ = feat_cols
        self.fitted_ = True
        return self

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        for c, v in self.impute_stats_.items():
            if c in df.columns:
                df[c] = df[c].fillna(v).astype("float64")
        return df

    def _ohe_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """OHE input as plain strings; NaN becomes the explicit level 'Unknown'
        so a missing category never produces a NaN feature."""
        out = pd.DataFrame(index=df.index)
        for c in OHE_COLS:
            out[c] = (df[c] if c in df.columns else pd.Series(pd.NA, index=df.index))
            out[c] = out[c].astype("object").where(out[c].notna(), "Unknown").astype(str)
        return out

    def _vif_drops(self, X: pd.DataFrame) -> list[str]:
        Xs = StandardScaler().fit_transform(X)
        drops = []
        for i, col in enumerate(X.columns):
            if col == "distance_km":
                continue  # keep the interpretable feature regardless of VIF
            try:
                # rank-deficiency warnings are expected with dropped-first OHE
                # columns; an unstable VIF is treated as "not a drop candidate".
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    v = variance_inflation_factor(Xs, i)
            except Exception:
                v = np.nan
            if np.isfinite(v) and v > self.vif_threshold:
                drops.append(col)
        return drops

    def _concat_ohe(self, df: pd.DataFrame) -> pd.DataFrame:
        arr = self.ohe_.transform(self._ohe_frame(df))
        ohe_df = pd.DataFrame(
            arr, columns=self.ohe_.get_feature_names_out(OHE_COLS), index=df.index
        )
        return pd.concat([df.drop(columns=OHE_COLS, errors="ignore"), ohe_df], axis=1)

    # ---- TRANSFORM: applies learned statistics. Fits nothing. ----
    def transform(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("transform() called before fit().")
        df = self._prepare(df_in)
        df = self._impute(df)

        for c, (lo, hi) in self.outlier_bounds_.items():
            if c in df.columns:
                df[c] = df[c].clip(lo, hi)

        df = self._concat_ohe(df)
        df = df.drop(columns=RAW_COORD_COLS, errors="ignore")

        # align to the learned feature set: add missing as 0, drop extras, fix order
        for c in self.feature_names_:
            if c not in df.columns:
                df[c] = 0.0
        X = df[self.feature_names_].astype("float64").copy()

        # Any NaN left here is a preprocessing bug, not data noise — fail loudly
        # rather than feed NaN through the scaler into the model.
        if X.isna().any().any():
            bad = X.columns[X.isna().any()].tolist()
            raise ValueError(f"NaN survived preprocessing in columns: {bad}")

        X[self.feature_names_] = self.scaler_.transform(X[self.feature_names_])

        if TARGET in df_in.columns:
            X[TARGET] = pd.to_numeric(df_in[TARGET], errors="coerce").to_numpy()
        return X

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)


def split_raw(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Single source of the train/test split. Called BEFORE any fit()."""
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
