"""AI/ML Task 2 demonstration: compare baseline and tuned housing predictions."""
from pathlib import Path
import joblib
import pandas as pd
import sklearn
import streamlit as st

st.set_page_config(page_title="California Housing | Task 2", page_icon="🏡", layout="wide")
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
MODEL_VERSION = "1.6.1"

@st.cache_resource
def load_models():
    base = Path(__file__).parent
    # Joblib loads Python pickle objects; only load locally supplied, trusted artifacts.
    baseline = joblib.load(base / "house_price_model.joblib")
    tuned = joblib.load(base / "task2_best_model.joblib")
    return baseline, tuned

st.title("🏡 California Housing | Model Comparison")
st.caption("MAINCRAFTS TECHNOLOGY · AI/ML INTERNSHIP · TASK 2")
st.write("Compare predictions from the original Linear Regression model (Task 1) and the cross-validated tuned Decision Tree (Task 2).")

if sklearn.__version__ != MODEL_VERSION:
    st.error(f"These saved models were trained with scikit-learn {MODEL_VERSION}; found {sklearn.__version__}. Install the pinned requirements.txt.")
    st.stop()

try:
    baseline, tuned = load_models()
except Exception as exc:
    st.error(f"Could not load models: {exc}")
    st.stop()

st.subheader("Held-out model performance")
st.caption("Same California Housing 80/20 split (random_state=42). Error values in USD are approximate; the dataset's target uses $100,000 units.")
metrics = pd.DataFrame([
    {"Model":"Linear Regression · Task 1", "Test RMSE ($)":74558, "Test R²":0.5758},
    {"Model":"Tuned Decision Tree · Task 2", "Test RMSE ($)":60220, "Test R²":0.7233},
])
st.dataframe(metrics, hide_index=True, use_container_width=True)
c1,c2,c3=st.columns(3)
c1.metric("Best test RMSE", "$60,220", "19.2% lower vs baseline")
c2.metric("Best test R²", "0.7233")
c3.metric("Selected model", "Tuned Decision Tree")

st.divider()
st.subheader("Try the models on the same housing characteristics")
with st.form("inputs"):
    c1, c2 = st.columns(2)
    with c1:
        income = st.number_input("Median household income ($)", min_value=1000, max_value=150000, value=35000, step=500)
        age = st.number_input("Median house age (years)", min_value=1, max_value=52, value=25)
        rooms = st.number_input("Average rooms per household", min_value=0.1, max_value=150.0, value=5.0, step=0.1)
        bedrooms = st.number_input("Average bedrooms per household", min_value=0.1, max_value=35.0, value=1.0, step=0.1)
    with c2:
        population = st.number_input("Block-group population", min_value=1, max_value=35682, value=1200)
        occupancy = st.number_input("Average occupants per household", min_value=0.1, max_value=1250.0, value=3.0, step=0.1)
        latitude = st.number_input("Latitude", min_value=32.54, max_value=41.95, value=34.0, step=0.01, format="%.2f")
        longitude = st.number_input("Longitude", min_value=-124.35, max_value=-114.31, value=-118.0, step=0.01, format="%.2f")
    run = st.form_submit_button("Predict with both models", type="primary", use_container_width=True)

if run:
    if bedrooms > rooms:
        st.error("Average bedrooms cannot exceed average rooms.")
    else:
        # scikit-learn's MedInc feature is measured in $10,000 units.
        row = pd.DataFrame([[income / 10000, age, rooms, bedrooms, population, occupancy, latitude, longitude]], columns=FEATURES)
        try:
            baseline_value = float(baseline.predict(row)[0]) * 100000
            tuned_value = float(tuned.predict(row)[0]) * 100000
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            left,right=st.columns(2)
            left.metric("Linear Regression prediction", f"${baseline_value:,.0f}")
            right.metric("Tuned Decision Tree prediction", f"${tuned_value:,.0f}")
            st.caption("These are two different estimates, not verified property prices. Test-set performance is aggregate and cannot determine which prediction is closer for this particular input.")
            if any(v < 0 or v > 500001 for v in [baseline_value,tuned_value]):
                st.warning("One or both estimates lie outside the observed capped target range; use additional caution.")

st.divider()
st.info("Educational demonstration only. The California Housing dataset is based on 1990 census block-group statistics, not current home listings. It predicts median block-group values rather than individual house prices.")
st.caption("Task 2 · N. Srinikethan · Original baseline and tuned decision-tree pipelines")
