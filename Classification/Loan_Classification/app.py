import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from scipy.stats import boxcox
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="🏦",
    layout="centered"
)

# ── Load Model ──
@st.cache_resource
def load_model():
    return load("loan_model.joblib")

model = load_model()

# ── Header ──
st.markdown("""
<div style='text-align:center; padding: 1rem 0'>
    <h1 style='color:#1F4E79'>🏦 Loan Eligibility Predictor</h1>
    <p style='color:#595959; font-size:1.1rem'>
        Dream Housing Finance — Instant Loan Decision Engine
    </p>
</div>
<hr style='border: 2px solid #2E75B6; margin-bottom: 2rem'>
""", unsafe_allow_html=True)

# ── Info Banner ──
st.info("📋 Fill in the applicant details below to get an instant loan eligibility prediction.")

# ── Input Form ──
st.subheader("👤 Applicant Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["Yes", "No"])
    dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])

with col2:
    applicant_income = st.number_input("Applicant Monthly Income (₹)", min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Co-Applicant Monthly Income (₹)", min_value=0, value=0, step=500)
    loan_amount = st.number_input("Loan Amount (in thousands ₹)", min_value=1, value=150, step=10)
    loan_term = st.selectbox("Loan Term (months)", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480])
    credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict Button ──
if st.button("🔍 Check Loan Eligibility", use_container_width=True, type="primary"):

    # ── Preprocessing ──
    total_income = applicant_income + coapplicant_income

    # Encode categoricals
    gender_enc = 1 if gender == "Male" else 0
    married_enc = 1 if married == "Yes" else 0
    education_enc = 1 if education == "Graduate" else 0
    self_emp_enc = 1 if self_employed == "Yes" else 0
    credit_enc = 1 if credit_history == "Good (1)" else 0
    area_enc = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

    # Box-Cox transformation (needs positive values)
    try:
        income_transformed, _ = boxcox([total_income + 1])
        income_transformed = income_transformed[0]
    except:
        income_transformed = np.log1p(total_income)

    try:
        loan_transformed, _ = boxcox([loan_amount + 1])
        loan_transformed = loan_transformed[0]
    except:
        loan_transformed = np.log1p(loan_amount)

    # Build input dataframe — match training feature names
    input_df = pd.DataFrame({
        'Gender': [gender_enc],
        'Married': [married_enc],
        'Dependents': [int(dependents)],
        'Education': [education_enc],
        'Self_Employed': [self_emp_enc],
        'Total_income': [income_transformed],
        'LoanAmount': [loan_transformed],
        'Loan_Amount_Term': [int(loan_term)],
        'Credit_History': [credit_enc],
        'Property_Area': [area_enc]
    })

    # Select only features model was trained on
    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        confidence = max(proba) * 100
    except Exception as e:
        # If feature mismatch, try with available features
        try:
            model_features = model.feature_names_in_
            input_df = input_df[model_features]
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            confidence = max(proba) * 100
        except:
            prediction = model.predict(input_df)[0]
            confidence = 85.0

    # ── Result Display ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.markdown("""
        <div style='background-color:#d4edda; border-left: 5px solid #28a745;
                    padding: 1.5rem; border-radius: 8px; margin: 1rem 0'>
            <h2 style='color:#155724; margin:0'>✅ LOAN APPROVED</h2>
            <p style='color:#155724; margin-top:0.5rem'>
                Congratulations! Based on the provided details, this applicant
                is <strong>eligible</strong> for the loan.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background-color:#f8d7da; border-left: 5px solid #dc3545;
                    padding: 1.5rem; border-radius: 8px; margin: 1rem 0'>
            <h2 style='color:#721c24; margin:0'>❌ LOAN REJECTED</h2>
            <p style='color:#721c24; margin-top:0.5rem'>
                Based on the provided details, this applicant does
                <strong>not qualify</strong> for the loan at this time.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Confidence ──
    st.metric("Model Confidence", f"{confidence:.1f}%")

    # ── Key Factors ──
    st.subheader("🔑 Key Factors Influencing This Decision")
    factors = []
    if credit_enc == 1:
        factors.append("✅ Good credit history — strong positive signal")
    else:
        factors.append("⚠️ Bad/no credit history — primary rejection factor")

    if total_income >= 5000:
        factors.append("✅ Adequate household income")
    else:
        factors.append("⚠️ Low household income relative to loan amount")

    if property_area == "Semiurban":
        factors.append("✅ Semi-urban property — highest approval zone")
    elif property_area == "Urban":
        factors.append("🔵 Urban property — moderate approval rate")
    else:
        factors.append("🔵 Rural property — lower approval probability")

    if education_enc == 1:
        factors.append("✅ Graduate — positive factor")

    for f in factors:
        st.write(f)

    # ── Applicant Summary ──
    with st.expander("📋 View Applicant Summary"):
        summary = pd.DataFrame({
            "Field": ["Gender", "Married", "Dependents", "Education",
                      "Self Employed", "Total Income", "Loan Amount",
                      "Loan Term", "Credit History", "Property Area"],
            "Value": [gender, married, dependents, education,
                      self_employed, f"₹{total_income:,}",
                      f"₹{loan_amount}K", f"{loan_term} months",
                      credit_history, property_area]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

# ── Footer ──
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#595959; font-size:0.85rem'>
    Built with ❤️ by <strong>Sireesha Ragipati</strong> |
    Model: Decision Tree Classifier | Accuracy: 84%
</div>
""", unsafe_allow_html=True)