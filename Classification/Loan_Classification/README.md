# 🏦 Loan Eligibility Prediction — End-to-End ML Classification Project

> **Automating loan approval decisions using Machine Learning** — comparing 8 algorithms to find the best model for Dream Housing Finance's real-time eligibility system.

---

## 🖥️ Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
&nbsp;&nbsp;

---

## 📌 Project Overview

**Dream Housing Finance** wants to automate its loan eligibility process based on customer details submitted in an online application. Instead of manual review, this ML system instantly predicts whether a loan should be **Approved ✅ or Rejected ❌**.

| Detail | Info |
|--------|------|
| **Domain** | Banking & Finance |
| **Problem Type** | Binary Classification |
| **Dataset Size** | 614 records, 13 features |
| **Best Model** | Decision Tree Classifier |
| **Best Test Accuracy** | **84%** |
| **Tools** | Python · Scikit-learn · XGBoost · Streamlit |

---

## 🎯 Business Objective

> Reduce manual loan review time and improve consistency by predicting loan approval based on applicant demographics, income, credit history, and property details.

---

## 🗺️ Project Workflow

```
Business Understanding
        ↓
Data Understanding & EDA
        ↓
Data Preparation (Cleaning → Encoding → Transformation)
        ↓
Model Building (8 Algorithms with GridSearchCV)
        ↓
Evaluation (Accuracy · CV Score · ROC-AUC)
        ↓
Best Model Selection → Saved with Joblib
        ↓
Deployed as Streamlit Web App
```

---

## 📖 Data Dictionary

| Column | Type | Description |
|--------|------|-------------|
| `Loan_ID` | Categorical | Unique loan reference (dropped) |
| `Gender` | Categorical | Male / Female |
| `Married` | Categorical | Yes / No |
| `Dependents` | Ordinal | 0, 1, 2, 3+ |
| `Education` | Categorical | Graduate / Not Graduate |
| `Self_Employed` | Categorical | Yes / No |
| `ApplicantIncome` | Continuous | Monthly income of applicant |
| `CoapplicantIncome` | Continuous | Monthly income of co-applicant |
| `LoanAmount` | Continuous | Loan amount requested (thousands) |
| `Loan_Amount_Term` | Discrete | Repayment term (months) |
| `Credit_History` | Binary | 1 = Good history, 0 = Bad |
| `Property_Area` | Categorical | Rural / Semi-Urban / Urban |
| `Loan_Status` | **Target** | Y = Approved, N = Rejected |

---

## 📊 Key EDA Insights

- 💳 **Credit History** is the strongest predictor — applicants with good credit get approved at **~80%+ rate**
- 🏘️ **Semi-urban** property area has the highest loan approval rate
- 🎓 **Graduates** are more likely to receive approval
- 👫 **Married applicants** show higher approval rates
- 📈 Both `Total_income` and `LoanAmount` are **right-skewed** → Box-Cox transformation applied

---

## 🛠️ Data Preparation Steps

| Step | Technique | Applied To |
|------|-----------|------------|
| Feature Drop | Remove `Loan_ID` | Unique ID — no predictive value |
| Feature Engineering | `ApplicantIncome + CoapplicantIncome → Total_income` | Captures household financial capacity |
| Missing Values | Mode imputation + Row drop | Categorical & critical numeric columns |
| Encoding | Label Encoding | All categorical features |
| Transformation | Box-Cox | `Total_income`, `LoanAmount` |
| Split | 80% Train / 20% Test | Full dataset |

---

## 🤖 Models Compared

| # | Model | CV Score | Test Accuracy |
|---|-------|----------|---------------|
| 1 | Logistic Regression | ~0.80 | ~0.80 |
| 2 | K-Nearest Neighbors | ~0.79 | ~0.77 |
| 3 | Support Vector Machine | ~0.81 | ~0.80 |
| 4 | **Decision Tree** ⭐ | **~0.83** | **~0.84** |
| 5 | Random Forest | ~0.82 | ~0.82 |
| 6 | AdaBoost | ~0.81 | ~0.80 |
| 7 | Gradient Boosting | ~0.82 | ~0.81 |
| 8 | XGBoost | ~0.82 | ~0.82 |

> 🏆 **Decision Tree** selected — best test accuracy + interpretable rules for business stakeholders

---

## 💡 SQL Highlights — Feature Selection Strategy

**Filter Method:** Dropped `Loan_ID` (zero variance, unique ID)

**Ensemble Method:** Used `feature_importances_` from Decision Tree to select only features with importance > 0:

```python
fea = pd.DataFrame(data=dt.feature_importances_,
                   index=x.columns, columns=["importance"])
dt_features = fea[fea["importance"] > 0].index.tolist()
```

**Hyperparameter Tuning with GridSearchCV:**
```python
param_grid = {"criterion": ["gini", "entropy"],
              "max_depth": list(range(1, 19))}
dt_hp = GridSearchCV(estimator, param_grid, cv=5, scoring="accuracy")
```

---

## 📂 Project Structure

```
Loan_Classification/
│
├── 📓 classification_Loan_all.ipynb   # Full ML notebook (EDA → Deployment)
├── 📊 LoanData.csv                    # Raw dataset (614 records)
├── 🤖 loan_model.joblib               # Saved Decision Tree model
├── 🌐 app.py                          # Streamlit web app
├── 📋 requirements.txt                # Python dependencies
└── 📝 README.md
```

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/SireeshaRagipati24/Machine-Learning.git
cd Machine-Learning/Classification/Loan_Classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

### 4. Open in browser
```
http://localhost:8501
```

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → Select your repo
4. Set **Main file path:** `Classification/Loan_Classification/app.py`
5. Click **Deploy** → Live in 2 minutes! 🎉

---

## 📈 Model Evaluation

```
Classification Report (Decision Tree):
              precision    recall  f1-score
           0       0.67      0.55      0.60
           1       0.87      0.92      0.89
    accuracy                           0.84

ROC-AUC Score: 0.73
5-Fold CV Score: 0.83
```

---

## 🔮 Future Improvements

- [ ] Handle class imbalance with **SMOTE**
- [ ] Add **SHAP values** for model explainability
- [ ] Try **stacking ensemble** for higher accuracy
- [ ] Add **loan amount recommendation** feature

---

## 🙋‍♀️ About Me

**Sireesha Ragipati** — Aspiring Data Analyst passionate about turning raw data into meaningful decisions.

[![GitHub](https://img.shields.io/badge/GitHub-SireeshaRagipati24-black?style=flat&logo=github)](https://github.com/SireeshaRagipati24)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/your-linkedin-here)

---

*⭐ If you found this helpful, give it a star!*
