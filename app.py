import streamlit as st
import pandas as pd
import joblib

# === 1. Load the trained models and scalers ===
final_model_ct = joblib.load("best_model_CT Index.pkl")
scaler_ct = joblib.load("scaler_CT Index.pkl")

final_model_rd = joblib.load("best_model_HWTT RD.pkl")
scaler_rd = joblib.load("scaler_HWTT RD.pkl")

# === 2. Define feature order (must match training) ===
feature_names = ["RAP ", "Binder Type", "Gyration", "Additives", "BSG", "VMA", "D/B", "AC, %"]

st.title("ML-Powered Prediction Tool for Mix Type IVS")

st.markdown("### Please enter mix design parameters:")

# === 3. User inputs via Web ===
rap = st.number_input("RAP contents (%)(input range: 0-20)", min_value=0.0, max_value=20.0, step=5.0)

binder = st.selectbox("Binder Type", options=[1, 2],
                      format_func=lambda x: "PG 70-28" if x == 1 else "PG 58E-34")

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

gyration = st.number_input("Design Gyration (input range: 50-80)", min_value=50.0, max_value=80.0, step=5.0)
bsg = st.number_input("BSG (input range: 2-3)", min_value=2.0, max_value=3.0, step=0.001)
vma = st.number_input("VMA (input range: 12-20)", min_value=12.0, max_value=20.0, step=0.1)
db_ratio = st.number_input("D/B ratio (input range: 0-2)", min_value=0.0, max_value=2.0, step=0.1)
ac = st.number_input("Asphalt Content, % (input range: 2-8.5)", min_value=2.0, max_value=8.5, step=0.1)

# === 4. Collect input ===
user_input = [rap, binder, gyration, additive, bsg, vma, db_ratio, ac]
manual_input = pd.DataFrame([user_input], columns=feature_names)

# === 5. Predict ===
if st.button("Predict Performance"):
    # --- CT Index ---
    manual_input_scaled_ct = scaler_ct.transform(manual_input)
    prediction_ct = final_model_ct.predict(manual_input_scaled_ct)

    # --- HWTT Rut Depth ---
    manual_input_scaled_rd = scaler_rd.transform(manual_input)
    prediction_rd = final_model_rd.predict(manual_input_scaled_rd)

    # --- Show results ---
    st.success(f"✅ Predicted CT Index: {prediction_ct[0]:.2f}")
    st.success(f"✅ Predicted HWTT Rut Depth: {prediction_rd[0]:.2f} mm")
