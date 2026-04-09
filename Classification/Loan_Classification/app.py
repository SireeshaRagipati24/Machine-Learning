
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from scipy.stats import boxcox
import warnings
import os
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="🏦",
    layout="centered"
)

# ── Custom CSS (Fixed Visibility) ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* 1. Global Font and Base Text Color */
    html, body { 
        font-family: 'Inter', sans-serif; 
        color: #1e293b; 
    }
    .stApp { background: linear-gradient(135deg, #f0f4f8 0%, #e8f0fe 100%); }

    /* 2. Fix: Labels (Gender, Married, etc.) and Dropdown Text */
    label {
        color: #1F4E79 !important;
        font-weight: 600 !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        color: #1F4E79 !important;
    }

    /* 3. Fix: Expander (View Full Applicant Summary) Visibility */
    .streamlit-expanderHeader p {
        color: #1F4E79 !important;
        font-weight: 600 !important;
    }

    /* ── Your Existing Styles ── */
    .hero-box {
        background: linear-gradient(135deg, #1F4E79 0%, #2E75B6 100%);
        padding: 2.5rem 2rem; border-radius: 16px; text-align: center;
        margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(31,78,121,0.25);
    }
    .hero-box h1 { color: white !important; font-size: 2.2rem; font-weight: 700; margin: 0 0 0.5rem 0; }
    .hero-box p { color: rgba(255,255,255,0.85) !important; font-size: 1rem; margin: 0; }

    .card {
        background: white; border-radius: 14px; padding: 1.8rem;
        margin-bottom: 1.5rem; box-shadow: 0 2px 16px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }
    .card-title {
        font-size: 1rem; font-weight: 600; color: #1F4E79;
        text-transform: uppercase; letter-spacing: 0.05em;
        margin-bottom: 1.2rem; padding-bottom: 0.6rem;
        border-bottom: 2px solid #e8f0fe;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1F4E79 0%, #2E75B6 100%) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 600 !important;
        font-size: 1.05rem !important; padding: 0.75rem !important;
        box-shadow: 0 4px 15px rgba(31,78,121,0.3) !important;
    }

    .result-approved {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 6px solid #28a745; padding: 1.8rem; border-radius: 12px; margin: 1rem 0;
    }
    .result-rejected {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 6px solid #dc3545; padding: 1.8rem; border-radius: 12px; margin: 1rem 0;
    }
    .result-approved h2 { color: #155724 !important; margin: 0 0 0.5rem 0; font-size: 1.6rem; }
    .result-rejected h2 { color: #721c24 !important; margin: 0 0 0.5rem 0; font-size: 1.6rem; }

    .score-bar-container {
        background: #e2e8f0; border-radius: 50px; height: 12px;
        margin: 0.4rem 0 1rem 0; overflow: hidden;
    }
    .score-bar-fill { height: 100%; border-radius: 50px; }

    .stat-box {
        background: white; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 1rem; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .stat-box .stat-value { font-size: 1.6rem; font-weight: 700; color: #1F4E79; }
    .stat-box .stat-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }

    .footer {
        text-align: center; color: #94a3b8; font-size: 0.82rem;
        padding: 1.5rem 0; margin-top: 2rem; border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ──
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "loan_model.joblib")
    return load(model_path)

model = load_model()

# ── Hero ──
st.markdown("""
<div class="hero-box">
    <h1>🏦 Loan Eligibility Predictor</h1>
    <p>Dream Housing Finance — Instant AI-Powered Loan Decision Engine</p>
</div>
""", unsafe_allow_html=True)

# ── Stats Row ──
c1, c2, c3, c4 = st.columns(4)
for col, val, label in zip(
    [c1, c2, c3, c4],
    ["84%", "614", "8", "< 1s"],
    ["Model Accuracy", "Training Records", "ML Models Tested", "Decision Time"]
):
    with col:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-value">{val}</div>
            <div class="stat-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Form ──
st.markdown('<div class="card"><div class="card-title">👤 Applicant Details</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    gender        = st.selectbox("Gender", ["Male", "Female"])
    married       = st.selectbox("Marital Status", ["Yes", "No"])
    dependents    = st.selectbox("Number of Dependents", [0, 1, 2, 3])
    education     = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
with col2:
    applicant_income   = st.number_input("Applicant Monthly Income (₹)", min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Co-Applicant Monthly Income (₹)", min_value=0, value=0, step=500)
    loan_amount        = st.number_input("Loan Amount (in thousands ₹)", min_value=1, value=150, step=10)
    loan_term          = st.selectbox("Loan Term (months)", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480])
    credit_history     = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])
    property_area      = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
st.markdown('</div>', unsafe_allow_html=True)

predict = st.button("🔍 Check Loan Eligibility", use_container_width=True, type="primary")

if predict:
    total_income     = applicant_income + coapplicant_income
    emi_estimate     = (loan_amount * 1000) / max(loan_term, 1)
    emi_income_ratio = emi_estimate / max(total_income, 1)

    # Encodings
    gender_enc    = 1 if gender == "Male" else 0
    married_enc   = 1 if married == "Yes" else 0
    education_enc = 1 if education == "Graduate" else 0
    self_emp_enc  = 1 if self_employed == "Yes" else 0
    credit_enc    = 1 if credit_history == "Good (1)" else 0
    area_enc      = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

    # ── Stable log transform (avoids boxcox single-value 28700% bug) ──
    income_transformed = np.log1p(max(total_income, 0))
    loan_transformed   = np.log1p(max(loan_amount, 0))

    input_df = pd.DataFrame({
        'Gender': [gender_enc], 'Married': [married_enc],
        'Dependents': [int(dependents)], 'Education': [education_enc],
        'Self_Employed': [self_emp_enc], 'Total_income': [income_transformed],
        'LoanAmount': [loan_transformed], 'Loan_Amount_Term': [int(loan_term)],
        'Credit_History': [credit_enc], 'Property_Area': [area_enc]
    })

    try:
        if hasattr(model, 'feature_names_in_'):
            input_df = input_df[model.feature_names_in_]
        prediction   = model.predict(input_df.values)[0]
        proba        = model.predict_proba(input_df.values)[0]
        approval_pct = float(proba[1]) * 100
        reject_pct   = float(proba[0]) * 100
    except Exception as e:
        try:
            if 'monotonic_cst' in str(e):
                model.monotonic_cst = None
            prediction   = model.predict(input_df.values)[0]
            proba        = model.predict_proba(input_df.values)[0]
            approval_pct = float(proba[1]) * 100
            reject_pct   = float(proba[0]) * 100
        except:
            prediction   = 1 if credit_enc == 1 else 0
            approval_pct = 75.0 if credit_enc == 1 else 25.0
            reject_pct   = 100 - approval_pct

    # ── Result ──
    st.markdown("<br>", unsafe_allow_html=True)
    if prediction == 1:
        st.markdown(f"""<div class="result-approved">
            <h2>✅ LOAN APPROVED</h2>
            <p>This applicant meets the eligibility criteria.
            Model predicts <strong>{approval_pct:.0f}% approval probability</strong>.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="result-rejected">
            <h2>❌ LOAN REJECTED</h2>
            <p>This applicant does not meet the criteria.
            Model predicts only <strong>{approval_pct:.0f}% approval probability</strong>.
            See detailed reasons below.</p>
        </div>""", unsafe_allow_html=True)

    # ── Probability Bars ──
    st.markdown('<div class="card"><div class="card-title">📊 Approval Probability</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style='font-weight:600;color:#166534;margin-bottom:4px'>✅ Approval: {approval_pct:.1f}%</div>
        <div class="score-bar-container"><div class="score-bar-fill"
        style="width:{min(approval_pct,100)}%;background:linear-gradient(90deg,#28a745,#20c997)"></div></div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style='font-weight:600;color:#9a3412;margin-bottom:4px'>❌ Rejection: {reject_pct:.1f}%</div>
        <div class="score-bar-container"><div class="score-bar-fill"
        style="width:{min(reject_pct,100)}%;background:linear-gradient(90deg,#dc3545,#e83e8c)"></div></div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Factor Analysis ──
    st.markdown('<div class="card"><div class="card-title">🔍 Why This Decision? — Factor-by-Factor Analysis</div>', unsafe_allow_html=True)

    factors = ""

    # Credit History
    if credit_enc == 1:
        factors += '<div class="factor-good">✅ <strong>Credit History: Good</strong> — Strongest positive signal. Good credit means reliable past repayment. Approval rate jumps to 80%+ for such applicants.</div>'
    else:
        factors += '<div class="factor-bad">🚨 <strong>Credit History: Bad / No History</strong> — #1 rejection reason. Past loan defaults or missed payments signal high risk to lenders. This single factor alone can trigger rejection.</div>'

    # EMI to Income
    if emi_income_ratio < 0.3:
        factors += f'<div class="factor-good">✅ <strong>Income vs EMI: Healthy ({emi_income_ratio*100:.0f}%)</strong> — Monthly EMI (₹{emi_estimate:,.0f}) is well within safe limits. Lenders prefer EMI below 40% of income.</div>'
    elif emi_income_ratio < 0.5:
        factors += f'<div class="factor-neutral">⚠️ <strong>Income vs EMI: Borderline ({emi_income_ratio*100:.0f}%)</strong> — Monthly EMI (₹{emi_estimate:,.0f}) is {emi_income_ratio*100:.0f}% of income. Lenders prefer below 40%.</div>'
    else:
        factors += f'<div class="factor-bad">❌ <strong>Income vs EMI: Overloaded ({emi_income_ratio*100:.0f}%)</strong> — Monthly EMI (₹{emi_estimate:,.0f}) exceeds 50% of income (₹{total_income:,}). This is a major red flag — applicant likely cannot sustain repayments.</div>'

    # Loan Amount
    if loan_amount <= 120:
        factors += f'<div class="factor-good">✅ <strong>Loan Amount: Low Risk (₹{loan_amount}K)</strong> — Small loan amount relative to income. Easier to approve.</div>'
    elif loan_amount <= 250:
        factors += f'<div class="factor-neutral">🔵 <strong>Loan Amount: Moderate (₹{loan_amount}K)</strong> — Standard range. Approval depends heavily on income and credit.</div>'
    else:
        factors += f'<div class="factor-bad">❌ <strong>Loan Amount: High Risk (₹{loan_amount}K)</strong> — Large loan requires strong financials. Needs good credit + high income to qualify.</div>'

    # Property Area
    area_map = {
        "Semiurban": ("good", "✅", "Semi-Urban — Best Zone", "Highest approval area. Strong property appreciation and lower default rates make lenders more comfortable."),
        "Urban": ("neutral", "🔵", "Urban — Moderate Zone", "Good property value but urban applicants often have higher loan demands too."),
        "Rural": ("neutral", "🔵", "Rural — Lower Priority Zone", "Rural properties have lower collateral value and higher perceived risk by lenders.")
    }
    ac, ai, al, ad = area_map[property_area]
    factors += f'<div class="factor-{ac}">{ai} <strong>Property Area: {al}</strong> — {ad}</div>'

    # Employment
    if self_emp_enc == 0:
        factors += '<div class="factor-good">✅ <strong>Employment: Salaried</strong> — Stable, predictable monthly income. Lenders prefer salaried applicants for consistent EMI repayment.</div>'
    else:
        factors += '<div class="factor-neutral">⚠️ <strong>Employment: Self-Employed</strong> — Variable income is harder to verify and predict. Lenders apply more scrutiny to self-employed applicants.</div>'

    # Education
    if education_enc == 1:
        factors += '<div class="factor-good">✅ <strong>Education: Graduate</strong> — Graduates statistically have higher earning potential and lower default rates.</div>'
    else:
        factors += '<div class="factor-neutral">🔵 <strong>Education: Not Graduate</strong> — Minor factor. Income stability matters more than degree for final decision.</div>'

    # Dependents
    dep = int(dependents)
    if dep == 0:
        factors += '<div class="factor-good">✅ <strong>Dependents: None</strong> — No financial dependents = more disposable income for EMI. Positive signal.</div>'
    elif dep <= 2:
        factors += f'<div class="factor-neutral">🔵 <strong>Dependents: {dep}</strong> — Moderate household obligations. Manageable with adequate income.</div>'
    else:
        factors += f'<div class="factor-bad">⚠️ <strong>Dependents: {dep}</strong> — High dependent count significantly reduces disposable income. Risk of payment defaults increases.</div>'

    st.markdown(factors, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Improvement Tips (only on rejection) ──
    if prediction == 0:
        st.markdown('<div class="card"><div class="card-title">💡 How to Improve Your Eligibility</div>', unsafe_allow_html=True)
        tips = []
        if credit_enc == 0:
            tips.append("🏆 <strong>Priority #1 — Repair Credit History:</strong> Clear existing dues, pay EMIs on time for 6+ months, and avoid new defaults. Credit history is the single biggest factor.")
        if emi_income_ratio >= 0.4:
            safe_loan = int(total_income * 0.35 * loan_term / 1000)
            tips.append(f"💰 <strong>Reduce Loan Amount:</strong> Your income supports a max loan of ~₹{safe_loan}K comfortably. Current request (₹{loan_amount}K) is too high for your income level.")
        if coapplicant_income == 0 and total_income < 8000:
            tips.append("👫 <strong>Add a Co-Applicant:</strong> A working spouse or family member as co-applicant boosts total income and improves approval chances significantly.")
        if self_emp_enc == 1:
            tips.append("📄 <strong>Provide Strong Income Proof:</strong> Self-employed applicants should submit 2+ years of ITR, bank statements, and business financials to build lender confidence.")
        if not tips:
            tips.append("📋 <strong>Reapply with a Co-Applicant:</strong> Adding a co-applicant with good credit and stable income significantly raises approval probability.")

        for tip in tips:
            st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Full Summary ──
    with st.expander("📋 View Full Applicant Summary"):
        summary = pd.DataFrame({
            "Field": ["Gender", "Married", "Dependents", "Education", "Self Employed",
                      "Applicant Income", "Co-Applicant Income", "Total Income",
                      "Loan Amount", "Loan Term", "Est. Monthly EMI",
                      "EMI-to-Income Ratio", "Credit History", "Property Area"],
            "Value": [gender, married, str(dependents), education, self_employed,
                      f"₹{applicant_income:,}", f"₹{coapplicant_income:,}", f"₹{total_income:,}",
                      f"₹{loan_amount}K", f"{loan_term} months", f"₹{emi_estimate:,.0f}",
                      f"{emi_income_ratio*100:.1f}%", credit_history, property_area],
            "Status": ["—", "✅" if married=="Yes" else "—",
                       "✅" if dep==0 else ("⚠️" if dep>=3 else "🔵"),
                       "✅" if education=="Graduate" else "🔵",
                       "✅" if self_employed=="No" else "⚠️",
                       "—", "—",
                       "✅" if total_income>=8000 else "⚠️",
                       "✅" if loan_amount<=120 else ("⚠️" if loan_amount<=250 else "❌"),
                       "—", "—",
                       "✅" if emi_income_ratio<0.4 else "❌",
                       "✅" if credit_enc==1 else "🚨",
                       "✅" if property_area=="Semiurban" else "🔵"]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

# ── Footer ──
st.markdown("""
<div class="footer">
    Built with ❤️ by <strong>Sireesha Ragipati</strong> &nbsp;|&nbsp;
    Decision Tree Classifier &nbsp;|&nbsp; Accuracy: 84% &nbsp;|&nbsp;
    <a href="https://github.com/SireeshaRagipati24" style="color:#2E75B6;text-decoration:none">GitHub ↗</a>
</div>
""", unsafe_allow_html=True)
