"""
APP.PY — Streamlit demo interface
===================================
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sys
import os

sys.path.append(os.path.dirname(__file__))
from generate_data import generate_signal, extract_features, SAMPLING_RATE

FAULT_NAMES = {0: "Normal", 1: "Inner Race Fault", 2: "Outer Race Fault", 3: "Ball Fault"}
FAULT_COLORS = {0: "#22c55e", 1: "#ef4444", 2: "#f97316", 3: "#eab308"}
FAULT_ADVICE = {
    0: "Equipment is operating normally. Continue routine monitoring.",
    1: "Inner race fault detected. Schedule bearing replacement within 2 weeks.",
    2: "Outer race fault detected. Inspect and replace bearing within 1 week.",
    3: "Ball fault detected. Monitor closely, replacement recommended within 3 weeks.",
}


@st.cache_resource
def load_model():
    model = joblib.load("outputs/model.pkl")
    feature_names = joblib.load("outputs/feature_names.pkl")
    return model, feature_names


def plot_signal(signal):
    t = np.linspace(0, len(signal) / SAMPLING_RATE, len(signal))
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(t * 1000, signal, linewidth=0.6, color="#4A90D9")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Acceleration (g)")
    ax.set_title("Vibration Signal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_fft(signal):
    from scipy.fft import fft, fftfreq
    n = len(signal)
    freqs = fftfreq(n, 1 / SAMPLING_RATE)[:n // 2]
    magnitude = np.abs(fft(signal))[:n // 2]

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(freqs, magnitude, linewidth=0.6, color="#7c85f5")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("FFT Spectrum")
    ax.set_xlim(0, 1000)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# --- UI ---
st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("Bearing Fault Detection")
st.caption("Vibration signal analysis for predictive maintenance using Random Forest")

try:
    model, feature_names = load_model()
except FileNotFoundError:
    st.error("Model not found. Run `python train.py` first.")
    st.stop()

st.sidebar.header("Simulate Sensor Input")
mode = st.sidebar.radio("Input mode", ["Simulate signal", "Enter features manually"])

if mode == "Simulate signal":
    fault_type = st.sidebar.selectbox(
        "Condition to simulate",
        options=[0, 1, 2, 3],
        format_func=lambda x: FAULT_NAMES[x]
    )
    noise = st.sidebar.slider("Noise level", 0.0, 1.0, 0.1)

    if st.sidebar.button("Run Analysis", type="primary"):
        signal = generate_signal(fault_type, 300)
        signal += np.random.normal(0, noise, len(signal))
        features = extract_features(signal)
        feature_vector = np.array([[features[f] for f in feature_names]])

        prediction = model.predict(feature_vector)[0]
        proba = model.predict_proba(feature_vector)[0]

        col1, col2 = st.columns([2, 1])

        with col1:
            st.pyplot(plot_signal(signal))
            st.pyplot(plot_fft(signal))

        with col2:
            color = FAULT_COLORS[prediction]
            st.markdown(f"""
            <div style="background:{color}22; border:2px solid {color};
                        border-radius:10px; padding:16px; margin-bottom:16px;">
                <div style="font-size:13px; color:{color}; font-weight:600; margin-bottom:4px;">
                    DIAGNOSIS
                </div>
                <div style="font-size:22px; font-weight:700; color:{color};">
                    {FAULT_NAMES[prediction]}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Confidence**")
            for i, (name, prob) in enumerate(zip(FAULT_NAMES.values(), proba)):
                st.progress(float(prob), text=f"{name}: {prob*100:.1f}%")

            st.info(FAULT_ADVICE[prediction])

            st.markdown("**Extracted Features**")
            feat_df = pd.DataFrame({
                "Feature": feature_names,
                "Value": [round(features[f], 4) for f in feature_names]
            })
            st.dataframe(feat_df, hide_index=True, use_container_width=True)

else:
    st.markdown("Enter feature values manually:")
    cols = st.columns(3)
    feature_vals = {}
    for i, feat in enumerate(feature_names):
        with cols[i % 3]:
            feature_vals[feat] = st.number_input(feat, value=0.0, format="%.4f")

    if st.button("Predict", type="primary"):
        feature_vector = np.array([[feature_vals[f] for f in feature_names]])
        prediction = model.predict(feature_vector)[0]
        proba = model.predict_proba(feature_vector)[0]
        color = FAULT_COLORS[prediction]
        st.markdown(f"### Prediction: :{color.replace('#','')}[{FAULT_NAMES[prediction]}]")
        for name, prob in zip(FAULT_NAMES.values(), proba):
            st.progress(float(prob), text=f"{name}: {prob*100:.1f}%")

st.divider()
col1, col2 = st.columns(2)
with col1:
    if os.path.exists("outputs/confusion_matrix.png"):
        st.image("outputs/confusion_matrix.png", caption="Confusion Matrix")
with col2:
    if os.path.exists("outputs/feature_importance.png"):
        st.image("outputs/feature_importance.png", caption="Feature Importance")
