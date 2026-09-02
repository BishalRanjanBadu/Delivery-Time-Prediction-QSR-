"""
Streamlit demo UI for the Zomato delivery-time model (Phase 2).

Thin front-end for single-order predictions — demos and manual checks. Production
traffic uses the FastAPI service; this calls the same transform-only predict path,
loading artifacts from S3, and shows the business layer (ETA band + window + SHAP).
"""
from __future__ import annotations

import os
import streamlit as st

import predict as P

st.set_page_config(page_title="Zomato Delivery-Time Predictor", page_icon="🛵")
st.title("Zomato Delivery-Time Predictor")

try:
    info = P.get_model_info()
    st.caption(
        f"Model {os.environ.get('MODEL_VERSION', 'unknown')} · "
        f"MAPE {info.get('mape', float('nan')):.1%} · trained {info.get('training_date', 'n/a')}"
    )
except Exception:
    st.caption(f"Model version: {os.environ.get('MODEL_VERSION', 'unknown')}")

c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Delivery person age", 18.0, 65.0, 30.0)
    rating = st.number_input("Delivery person rating", 1.0, 5.0, 4.7, step=0.1)
    distance = st.number_input("Distance (km)", 0.0, 50.0, 9.0)
    hour = st.slider("Order hour", 0, 23, 19)
    weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Fog", "Sandstorms", "Stormy", "Windy"])
with c2:
    traffic = st.selectbox("Traffic density", ["Low", "Medium", "High", "Jam"])
    vehicle = st.selectbox("Vehicle", ["motorcycle", "scooter", "electric_scooter", "bicycle"])
    order_type = st.selectbox("Order type", ["Snack", "Meal", "Drinks", "Buffet"])
    festival = st.selectbox("Festival", ["No", "Yes"])
    city = st.selectbox("City type", ["Urban", "Metropolitian", "Semi-Urban"])
    multi = st.selectbox("Multiple deliveries", [0.0, 1.0, 2.0, 3.0])

if st.button("Predict delivery time"):
    record = {
        "Delivery_person_Age": age, "Delivery_person_Ratings": rating,
        "Restaurant_latitude": 0.0, "Restaurant_longitude": 0.0,
        "Delivery_location_latitude": 0.0, "Delivery_location_longitude": 0.0,
        "Order_Date": "15-03-2022", "Time_Orderd": f"{hour:02d}:00",
        "Time_Order_picked": f"{hour:02d}:10", "Weather_conditions": weather,
        "Road_traffic_density": traffic, "Vehicle_condition": 1,
        "Type_of_order": order_type, "Type_of_vehicle": vehicle,
        "multiple_deliveries": multi, "Festival": festival, "City": city,
        "distance_km": distance,
    }
    try:
        res = P.predict_records([record])[0]
        lo, hi = res["delivery_window_min"]
        st.metric("Estimated delivery time", f"{res['predicted_time_min']:.0f} min",
                  help=f"Likely window {lo:.0f}–{hi:.0f} min")
        st.write(f"**ETA band:** {res['eta_band']}  ·  **Window:** {lo:.0f}–{hi:.0f} min")
        if res.get("shap_explanation"):
            st.write("**Top drivers of this prediction:**")
            st.bar_chart(res["shap_explanation"])
    except Exception as e:  # noqa: BLE001
        st.error(f"Prediction failed: {e}")
