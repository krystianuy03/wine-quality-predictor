import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
@st.cache_resource
def load_model():
    return joblib.load("models/wine_pipeline.joblib")

model = load_model()

# App Title
st.title("🍷 Wine Quality Predictor")
st.write("Predict if a wine is **GOOD** (quality ≥ 7) or **NOT GOOD** (quality < 7).")

# Define input fields for chemical attributes
st.subheader("Enter Wine Sample Attributes")

fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.7)
citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.0)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=20.0, value=1.9)
chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, value=11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=300.0, value=34.0)
density = st.number_input("Density", min_value=0.9900, max_value=1.0050, value=0.9978, format="%.4f")
pH = st.number_input("pH", min_value=2.0, max_value=5.0, value=3.51)
sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.56)
alcohol = st.number_input("Alcohol", min_value=8.0, max_value=15.0, value=9.4)

# Collect inputs into a DataFrame
input_data = pd.DataFrame({
    "fixed acidity": [fixed_acidity],
    "volatile acidity": [volatile_acidity],
    "citric acid": [citric_acid],
    "residual sugar": [residual_sugar],
    "chlorides": [chlorides],
    "free sulfur dioxide": [free_sulfur_dioxide],
    "total sulfur dioxide": [total_sulfur_dioxide],
    "density": [density],
    "pH": [pH],
    "sulphates": [sulphates],
    "alcohol": [alcohol]
})

# Predict button
if st.button("Predict Wine Quality"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success(f"✅ This wine is predicted to be GOOD with confidence {proba[1]:.2f}")
    else:
        st.error(f"❌ This wine is predicted to be NOT GOOD with confidence {proba[0]:.2f}")
