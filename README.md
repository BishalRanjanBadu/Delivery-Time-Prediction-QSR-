"""
Feature engineering for QSR delivery-time prediction.

Logic preserved from the original notebook (verified clean — no target
leakage: order_prep_time is built from Time_Orderd/Time_Order_picked,
which are both upstream of Time_taken (min), and the train/test split
is a random split, which is correct here since each row is an
independent delivery, not a time series).

Framing note: `order_prep_time` is the single strongest predictor, but
it reflects the *actual* prep-to-pickup duration -- only knowable once
the rider has picked up the order. If this model is meant to quote an
ETA to the customer at order placement (before pickup), this feature
isn't actually available yet at that moment; treat this as an ETA
*refresh* once an order is picked up, not a pre-dispatch quote, unless
order_prep_time is replaced with a predicted/estimated value.
"""
import pandas as pd

TRAFFIC_MAP = {'Low': 1, 'Medium': 2, 'High': 3, 'Jam': 4}

CATEGORICAL_ONE_HOT_COLS = ['Weather_conditions', 'Type_of_order', 'Type_of_vehicle', 'City']

DROP_BEFORE_MODELING = [
    'Time_taken (min)', 'Order_Date', 'Time_Orderd', 'Time_Order_picked',
    'Time_Orderd_dt', 'Time_Order_picked_dt', 'Road_traffic_density',
]

TARGET = 'Time_taken (min)'


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['Time_Orderd_dt'] = pd.to_datetime(df['Time_Orderd'], errors='coerce')
    df['Time_Order_picked_dt'] = pd.to_datetime(df['Time_Order_picked'], errors='coerce')

    df['order_hour'] = df['Time_Orderd_dt'].dt.hour
    df['pickup_hour'] = df['Time_Order_picked_dt'].dt.hour
    df['order_prep_time'] = (df['Time_Order_picked_dt'] - df['Time_Orderd_dt']).dt.total_seconds() / 60

    for col in ['order_hour', 'pickup_hour', 'order_prep_time']:
        df[col] = df[col].fillna(df[col].median())

    df['is_peak_hour'] = df['order_hour'].apply(lambda x: 1 if (11 <= x <= 14) or (18 <= x <= 22) else 0)
    df['traffic_level'] = df['Road_traffic_density'].map(TRAFFIC_MAP)
    df['distance_traffic'] = df['distance_km'] * df['traffic_level']

    df = pd.get_dummies(df, columns=CATEGORICAL_ONE_HOT_COLS, drop_first=True)
    df['Festival'] = df['Festival'].map({'Yes': 1, 'No': 0})

    df['Delivery_person_Age'] = df['Delivery_person_Age'].fillna(df['Delivery_person_Age'].median())
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].fillna(df['Delivery_person_Ratings'].median())

    return df


def build_model_matrix(df: pd.DataFrame):
    y = df[TARGET]
    X = df.drop(columns=DROP_BEFORE_MODELING)
    return X, y
