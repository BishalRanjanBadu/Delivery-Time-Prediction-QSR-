"""
Data loading and cleaning for the QSR delivery-time dataset.

Extracted from the original notebook. The leakage-avoidance logic
(dropping ID, Delivery_person_ID, delivery_speed) was already correct
in the original and is preserved as-is.
"""
import pandas as pd

LEAKY_OR_ID_COLS = ['ID', 'Delivery_person_ID', 'delivery_speed']


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns and `delivery_speed`, which is derived from
    the outcome (distance/time) and would leak the target."""
    return df.drop(columns=LEAKY_OR_ID_COLS)


def load_and_clean(path: str) -> pd.DataFrame:
    df = load_raw(path)
    return drop_leakage_columns(df)
