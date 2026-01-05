import streamlit as st
import pandas as pd
import joblib
from sqlalchemy import create_engine

# ----------------------------------
# Page configuration
# ----------------------------------
st.set_page_config(
    page_title="Bangalore Airbnb Dynamic Pricing",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Bangalore Airbnb Dynamic Pricing")
st.write("ML-powered Airbnb price recommendation with business logic")

# ----------------------------------
# Load trained model & feature list
# ----------------------------------
model = joblib.load("pricing_model.pkl")
model_features = joblib.load("model_features.pkl")

# ----------------------------------
# Database connection (USE YOUR REAL PASSWORD)
# ----------------------------------
ref_df = pd.read_csv("reference_data.csv")


# ----------------------------------
# Sidebar inputs
# ----------------------------------
st.sidebar.header("Listing Details")

accommodates = st.sidebar.slider("Number of Guests", 1, 10, 2)

room_type = st.sidebar.selectbox(
    "Room Type",
    sorted(ref_df["room_type"].dropna().unique())
)

neighbourhood = st.sidebar.selectbox(
    "Neighbourhood",
    sorted(ref_df["neighbourhood"].dropna().unique())
)

superhost_flag = st.sidebar.selectbox(
    "Superhost",
    ["No", "Yes"]
)

day_type = st.sidebar.selectbox(
    "Day Type",
    ["Weekday", "Weekend"]
)

demand_level = st.sidebar.selectbox(
    "Demand Level",
    ["Low", "Medium", "High"]
)

# ----------------------------------
# Prepare input dataframe
# ----------------------------------
input_data = {
    "accommodates": accommodates,
    "price_per_guest": 0,  # placeholder (model learned pattern)
    "superhost_flag": 1 if superhost_flag == "Yes" else 0
}

input_df = pd.DataFrame([input_data])

# Add all missing model columns
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

# Activate selected categorical columns
room_col = f"room_type_{room_type}"
neigh_col = f"neighbourhood_{neighbourhood}"

if room_col in input_df.columns:
    input_df[room_col] = 1

if neigh_col in input_df.columns:
    input_df[neigh_col] = 1

# Reorder columns exactly as model expects
input_df = input_df[model_features]

# ----------------------------------
# Prediction + Business Logic
# ----------------------------------
if st.button("Predict Price 💰"):
    base_price = model.predict(input_df)[0]

    adjusted_price = base_price

    # Weekend premium
    if day_type == "Weekend":
        adjusted_price *= 1.10

    # Superhost premium
    if superhost_flag == "Yes":
        adjusted_price *= 1.05

    # Demand-based adjustment
    if demand_level == "High":
        adjusted_price *= 1.15
    elif demand_level == "Low":
        adjusted_price *= 0.95

    # Minimum price floor
    adjusted_price = max(adjusted_price, 1000)

    st.success(f"💵 Recommended Price: ₹ {round(adjusted_price, 2)} per night")

    st.caption(
        "Price = ML prediction + weekend, demand & host-quality adjustments"
    )
