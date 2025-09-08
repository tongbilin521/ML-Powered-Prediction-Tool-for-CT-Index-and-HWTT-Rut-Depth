import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# ======== 公共设置 ========
feature_cols = ["RAP", "Binder Type", "Gyration", "Additives", "BSG", "VMA", "D/B", "AC, %"]
plt.rcParams.update({'font.size': 24})  # 调整全局字体

# 自定义 percentage error 函数
def percentage_error_func(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

# 批量评估函数
def evaluate_and_plot(ax, model_path, scaler_path, file, target_col, title, x_label, y_label):
    final_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    df = pd.read_excel(file)
    df.columns = df.columns.str.strip()

    X_new = df[feature_cols]
    y_actual = df[target_col]

    # === 自动对齐列名（保证和训练时完全一致） ===
    scaler_cols = list(scaler.feature_names_in_)          # 训练时的原始列名
    col_map = {c.strip(): c for c in scaler_cols}         # 去空格映射回原始列名

    # Excel 列名去空格再重命名成 scaler 的原始列名
    X_new.columns = [c.strip() for c in X_new.columns]
    X_new = X_new.rename(columns=col_map)

    # 强制按照 scaler 的列顺序排列
    X_new = X_new[scaler_cols]


    X_new_scaled = scaler.transform(X_new)
    y_pred = final_model.predict(X_new_scaled)

    # 计算指标
    r2 = r2_score(y_actual, y_pred)
    perc_error = percentage_error_func(y_actual, y_pred)
    accuracy = 1 - perc_error

    ax.scatter(y_actual, y_pred, c="blue", alpha=0.6, edgecolors="k", s=120, label="Data points")
    ax.plot([y_actual.min(), y_actual.max()],
             [y_actual.min(), y_actual.max()],
             'r--', lw=3, label="Ideal fit (y=x)")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # 在图内写 R² 和 Accuracy
    ax.text(0.05, 0.95,
            f"R² = {r2:.2f}\nAcc = {accuracy:.2f}",
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    ax.legend()
    ax.grid(True)

# ======== Streamlit 应用 ========
st.title("ML-Powered Prediction Tool for Mix Type IVS")

mode = st.radio("Select Prediction Mode:", ["Single Prediction", "Batch Prediction"])

# --- 单一样本预测 ---
if mode == "Single Prediction":
    st.subheader("🔹 Single Prediction (Manual Input)")
    # 输入表单
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

    # 收集输入
    manual_input = pd.DataFrame([[rap, binder, gyration, additive, bsg, vma, db_ratio, ac]],
                                columns=feature_cols)

    if st.button("Predict"):
        # 加载模型
        model_ct = joblib.load("best_model_CT Index.pkl")
        scaler_ct = joblib.load("scaler_CT Index.pkl")
        model_rd = joblib.load("best_model_HWTT RD.pkl")
        scaler_rd = joblib.load("scaler_HWTT RD.pkl")

        # CT Index 预测
        manual_scaled_ct = scaler_ct.transform(manual_input)
        pred_ct = model_ct.predict(manual_scaled_ct)

        # HWTT RD 预测
        manual_scaled_rd = scaler_rd.transform(manual_input)
        pred_rd = model_rd.predict(manual_scaled_rd)

        st.success(f"✅ Predicted CT Index: {pred_ct[0]:.2f}")
        st.success(f"✅ Predicted HWTT Rut Depth: {pred_rd[0]:.2f} mm")

# --- 批量预测 ---
else:
    st.subheader("🔹 Batch Prediction (Upload Excel Files)")
    file_ct = st.file_uploader("Upload CT Index Excel", type=["xlsx"])
    file_rd = st.file_uploader("Upload HWTT RD Excel", type=["xlsx"])

    if file_ct is not None and file_rd is not None:
        fig, axes = plt.subplots(1, 2, figsize=(26, 12))

        evaluate_and_plot(
            axes[0],
            model_path="best_model_CT Index.pkl",
            scaler_path="scaler_CT Index.pkl",
            file=file_ct,
            target_col="Avg. CT index",
            title="Predicted vs Actual CT index",
            x_label="Actual CT index",
            y_label="Predicted CT index"
        )

        evaluate_and_plot(
            axes[1],
            model_path="best_model_HWTT RD.pkl",
            scaler_path="scaler_HWTT RD.pkl",
            file=file_rd,
            target_col="Avg. Rut Depth",
            title="Predicted vs Actual HWTT RD",
            x_label="Actual HWTT RD",
            y_label="Predicted HWTT RD"
        )

        st.pyplot(fig)

