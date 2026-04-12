# 🏦 Credit Risk Modelling — End-to-End ML Classification Project

> **Predicting whether a loan applicant is a good or bad credit risk** — using ensemble ML models on the German Credit Dataset to help banks make smarter, data-driven lending decisions.

---

## 📌 Project Overview

| Detail | Info |
|--------|------|
| **Domain** | Banking & Finance / Risk Analytics |
| **Dataset** | German Credit Data — 1,000 loan applicants |
| **Problem Type** | Binary Classification (Good Risk vs Bad Risk) |
| **Target Variable** | `Risk` — good (0) or bad (1) |
| **Best Model** | Extra Trees Classifier (~76% accuracy) |
| **Algorithms Compared** | Decision Tree · Random Forest · Extra Trees · XGBoost |
| **Key Challenge** | Class Imbalance — 70% good, 30% bad risk |

---

## 🎯 Business Objective

> **Can we predict whether a loan applicant will default** — before the bank approves the loan?

Banks face two costly mistakes:
- ❌ **Approve bad loans** → Non-Performing Assets (NPAs), financial loss
- ❌ **Reject good customers** → Lost revenue, reputation damage

A smart ML model helps banks make **accurate, fair, and explainable credit decisions at scale**.

---

## ⚠️ Why Credit Risk is a Hard ML Problem

| Challenge | Description | Our Solution |
|-----------|-------------|-------------|
| **Class Imbalance** | 70% good, 30% bad — naive model predicts "good" always | `class_weight='balanced'` + `scale_pos_weight` |
| **High Stakes** | False Negatives = approved bad loans = real money lost | Prioritise Recall for bad risk class |
| **Mixed Features** | Numeric + categorical data | LabelEncoder per column, saved for deployment |
| **Missing Values** | 18% Savings, 39% Checking account missing | Drop rows (financial status can't be safely imputed) |

---

## 📖 Dataset — Feature Dictionary

| Feature | Type | Description |
|---------|------|-------------|
| `Age` | Numerical | Applicant age in years |
| `Sex` | Categorical | male / female |
| `Job` | Ordinal | 0=unskilled non-resident → 3=highly skilled |
| `Housing` | Categorical | own / free / rent |
| `Saving accounts` | Categorical | little / moderate / quite rich / rich |
| `Checking account` | Categorical | little / moderate / rich |
| `Credit amount` | Numerical | Loan amount requested (Deutsche Mark) |
| `Duration` | Numerical | Loan repayment term (months) |
| `Purpose` | Categorical | car / furniture / education / business / etc. |
| `Risk` | **Target** | good = low risk ✅ / bad = high risk ❌ |

---

## 🗺️ Project Workflow

```
Load German Credit Dataset (1,000 records)
              ↓
Data Understanding (shape, dtypes, missing values)
              ↓
Data Cleaning → Drop missing rows (1000 → 603 clean records)
              ↓
EDA
  → Univariate: histograms, boxplots, countplots
  → Bivariate: all features vs Risk target
  → Correlation heatmap
  → Scatter, violin, pivot table analysis
              ↓
Feature Selection (8 most impactful features)
              ↓
Label Encoding → categoricals → numeric (saved as .pkl)
              ↓
Stratified Train-Test Split (80/20)
              ↓
Train 4 Models with GridSearchCV (5-fold CV)
  → Decision Tree
  → Random Forest
  → Extra Trees  ← Best
  → XGBoost
              ↓
Compare: Accuracy + ROC-AUC + Confusion Matrix
              ↓
Feature Importance Analysis
              ↓
Save Model + All Encoders (joblib)
```

---

## 📊 Key EDA Insights

| Insight | Finding |
|---------|---------|
| **Credit amount** | Bad risk applicants request significantly higher amounts |
| **Duration** | Bad risk loans have longer repayment terms |
| **Saving accounts** | Little/no savings → highest bad risk proportion |
| **Checking account** | Little checking balance → strongest single risk predictor |
| **Age** | Younger applicants lean toward higher risk |
| **Purpose** | Vacation loans show highest bad risk rate |

---

## 🤖 Models Compared

| Model | Test Accuracy | Key Strength |
|-------|-------------|-------------|
| Decision Tree | ~72% | Interpretable — single rule tree |
| Random Forest | ~74% | Stable, low variance |
| **Extra Trees** ⭐ | **~76%** | More randomness → best generalisation |
| XGBoost | ~75% | Sequential error correction |

> 🏆 **Extra Trees selected** — best accuracy, fastest training, excellent generalisation.

---

## 💡 Class Imbalance Handling

```python
# Tree models: penalise bad-risk errors more heavily
model = ExtraTreesClassifier(class_weight='balanced')

# XGBoost: explicit positive class weight
scale_pos_weight = count(good) / count(bad)   # ~2.3
xgb = XGBClassifier(scale_pos_weight=scale_pos_weight)

# Split: preserve class ratio in both sets
train_test_split(X, y, stratify=y)
```

Without this, models predict "good" for everyone → 70% accuracy but miss ALL bad loans!

---

## 📦 Saved Deployment Files

| File | Contents |
|------|---------|
| `extra_trees_credit_model.pkl` | Trained best model |
| `Sex_encoder.pkl` | LabelEncoder for Sex |
| `Housing_encoder.pkl` | LabelEncoder for Housing |
| `Saving accounts_encoder.pkl` | LabelEncoder for Savings |
| `Checking account_encoder.pkl` | LabelEncoder for Checking |
| `target_encoder.pkl` | Decode predictions → good/bad |

---

## 📂 Project Structure

```
Credit_Risk_Modelling/
│
├── 📓 Credit_Risk_Modelling.ipynb     # Full notebook (end-to-end)
├── 📊 german_credit_data.csv          # Raw dataset (1000 records)
├── 🤖 extra_trees_credit_model.pkl    # Best model
├── 🔤 *_encoder.pkl                   # Per-column LabelEncoders
└── 📝 README.md
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/SireeshaRagipati24/Machine-Learning.git
cd Machine-Learning/Classification/Credit_Risk_Modelling

pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib

jupyter notebook Credit_Risk_Modelling.ipynb
```

---

## 🔮 Predict on New Applicant

```python
import joblib, pandas as pd

model  = joblib.load('extra_trees_credit_model.pkl')
le_sex = joblib.load('Sex_encoder.pkl')
le_hous= joblib.load('Housing_encoder.pkl')
le_sav = joblib.load('Saving accounts_encoder.pkl')
le_chk = joblib.load('Checking account_encoder.pkl')

new_applicant = pd.DataFrame([{
    'Age': 35, 'Sex': le_sex.transform(['male'])[0],
    'Job': 2, 'Housing': le_hous.transform(['own'])[0],
    'Saving accounts': le_sav.transform(['moderate'])[0],
    'Checking account': le_chk.transform(['little'])[0],
    'Credit amount': 5000, 'Duration': 24
}])

risk = model.predict(new_applicant)[0]
prob = model.predict_proba(new_applicant)[0]
print(f'Risk: {"BAD ❌" if risk==1 else "GOOD ✅"}  |  Confidence: {max(prob)*100:.1f}%')
```

---

## 💡 Key ML Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| Class Imbalance | `class_weight='balanced'` + `scale_pos_weight` |
| Stratified Split | `stratify=y` — preserves 70/30 ratio |
| Label Encoding | Per-column encoders saved as `.pkl` |
| GridSearchCV | 5-fold CV for all 4 models |
| ROC-AUC | All models compared on AUC curve |
| Feature Importance | Ensemble method — which features drive risk |
| Model Persistence | joblib — model + encoders for deployment |
| Reusable Pipeline | `train_model()` — one function for all algorithms |

---

## 🚀 Future Improvements

- [ ] **SMOTE oversampling** — better imbalance handling
- [ ] **SHAP values** — explainable predictions (banking regulatory requirement)
- [ ] **Streamlit web app** — input applicant details → instant risk score
- [ ] **Threshold tuning** — optimise for Recall on bad risk
- [ ] **Logistic Regression** — interpretable coefficient-based baseline

---

## 🙋‍♀️ About Me

**Sireesha Ragipati** — Aspiring Data Analyst | Finance Domain Enthusiast

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/sireesha-ragipati-269a10244/)

---

*⭐ If you found this helpful, give it a star!*
