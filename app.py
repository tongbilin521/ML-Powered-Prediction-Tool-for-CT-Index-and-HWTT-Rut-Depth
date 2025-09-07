import streamlit as st
import pandas as pd
import joblib

# === 1. Load the trained model and scaler ===
final_model = joblib.load("best_model_CT Index.pkl")
scaler = joblib.load("scaler_CT Index.pkl")

# === 2. Define feature order (must match training) ===
feature_names = ["RAP ", "Binder Type", "Gyration","Additives", "BSG", "VMA", "D/B", "AC, %"]

st.title("ML-Powered Prediction Tool for Mix Type IVS (CT Index)")

# === 3. User inputs via Web ===
rap = st.number_input("RAP contents (%)", min_value=0.0, max_value=20.0, step=0.5)

binder = st.selectbox("Binder Type", options=[1, 2],
                      format_func=lambda x: "PG 70-28" if x==1 else "PG 58E-34")

additives_options = {
    1: "None",
    2: "0.5% Rediset LQ",
    3: "0.3% Evotherm",
    4: "0.5% Evotherm",
    5: "0.75% Evotherm",
    6: "1.0 - 1.5% water",
    7: "1.5 - 2.0% water",
    8: "0.1% Zycotherm",
    9: "0.5% SonneWarmix",
    10: "1-1.5% Double Barrel Green Foaming"
}
additive = st.selectbox("Additives", options=list(additives_options.keys()),
                        format_func=lambda x: additives_options[x])

gyration = st.number_input("Design Gyration", min_value=30.0, max_value=100.0, step=1.0)
bsg = st.number_input("BSG", min_value=1.5, max_value=3.0, step=0.01)
vma = st.number_input("VMA", min_value=5.0, max_value=25.0, step=0.1)
db_ratio = st.number_input("D/B ratio", min_value=0.5, max_value=2.0, step=0.01)
ac = st.number_input("Asphalt Content (AC, %)", min_value=2.0, max_value=10.0, step=0.1)

# === 4. Collect input ===
user_input = [rap, binder, gyration, additive, bsg, vma, db_ratio, ac]
manual_input = pd.DataFrame([user_input], columns=feature_names)

# === 5. Predict ===
if st.button("Predict CT Index"):
    manual_input_scaled = scaler.transform(manual_input)
    prediction = final_model.predict(manual_input_scaled)
    st.success(f"✅ Predicted CT Index: {prediction[0]:.2f}")
