"""Wine Cultivar Predictor - an interactive Streamlit front-end for a trained ML model."""
import json
import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.joblib"
META_PATH = BASE_DIR / "model_meta.json"

st.set_page_config(page_title="Wine Cultivar Predictor", page_icon="🍷", layout="centered")

# Friendly labels + units shown beside each slider
LABELS = {
    "alcohol": ("Alcohol", "% vol"),
    "malic_acid": ("Malic acid", "g/L"),
    "total_phenols": ("Total phenols", "index"),
    "flavanoids": ("Flavanoids", "index"),
    "color_intensity": ("Colour intensity", "index"),
    "proline": ("Proline", "mg/L"),
}


@st.cache_resource
def load_model():
    """Load the trained pipeline once; Streamlit reuses it on every rerun."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_meta():
    with open(META_PATH) as fh:
        return json.load(fh)


st.title("🍷 Wine Cultivar Predictor")
st.write(
    "Move the sliders to describe a wine's chemical analysis and the model will "
    "estimate which of three grape cultivars it most likely came from."
)

# ---- Step 7: friendly handling of missing files (no raw error trace) ----
try:
    model = load_model()
    meta = load_meta()
except FileNotFoundError:
    st.error(
        "The model files are missing. Run `python train_model.py` first to create "
        "`model.joblib` and `model_meta.json`, then restart the app."
    )
    st.stop()
except Exception:
    st.error("The model could not be loaded. Please re-train it and try again.")
    st.stop()

FEATURES = meta["features"]  # same order the model was trained with
CLASSES = meta["classes"]

# ---- Step 4: input widgets with sensible min / max / default ----
st.subheader("1. Describe the wine")
left, right = st.columns(2)
values = {}
for i, feat in enumerate(FEATURES):
    rng = meta["ranges"][feat]
    label, unit = LABELS.get(feat, (feat, ""))
    step = 1.0 if rng["max"] > 100 else 0.01
    with (left if i % 2 == 0 else right):
        values[feat] = st.slider(
            f"{label} ({unit})",
            min_value=float(rng["min"]),
            max_value=float(rng["max"]),
            value=float(rng["default"]),
            step=step,
        )

# ---- Step 5: prediction on button press ----
st.subheader("2. Get a prediction")
if st.button("Predict", type="primary"):
    # Validate input before predicting
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in values.values()):
        st.warning("Some inputs are empty or invalid. Please check the sliders and try again.")
        st.stop()

    try:
        row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
        proba = model.predict_proba(row)[0]
    except Exception:
        st.error("Something went wrong while making the prediction. Please adjust the inputs and retry.")
        st.stop()

    best = int(proba.argmax())
    c1, c2 = st.columns(2)
    c1.metric("Most likely cultivar", CLASSES[best])
    c2.metric("Model confidence", f"{proba[best]:.0%}")

    st.write("**Probability for each cultivar**")
    chart_df = pd.DataFrame({"Probability": proba}, index=CLASSES)
    st.bar_chart(chart_df)

    if proba[best] < 0.6:
        st.info("The model is not very sure about this combination of values.")

# ---- Step 6: explain the result ----
st.divider()
st.caption(
    f"About this app: a Random Forest classifier trained on the classic UCI Wine dataset "
    f"({meta['n_samples']} wines, 3 cultivars from one region of Italy) using {len(FEATURES)} chemical "
    f"measurements. Accuracy on held-out test data: {meta['accuracy']:.0%}. "
    "This is an educational example and not advice for buying, selling or judging wine."
)
