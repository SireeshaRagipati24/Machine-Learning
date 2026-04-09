# 🎓 University Admission Prediction — End-to-End Regression Project

> **Predicting a student's chances of getting into graduate school** — comparing 9 regression algorithms to find the most accurate and interpretable admission prediction model.

---

## 📌 Project Overview

| Detail | Info |
|--------|------|
| **Domain** | Education / Predictive Analytics |
| **Problem Type** | Regression (predict continuous probability 0–1) |
| **Dataset** | Graduate Admissions — 500 student records |
| **Target Variable** | `Chance of Admit` (0.0 → 1.0) |
| **Best Model** | Linear Regression |
| **Best R² Score** | ~82–85% |
| **Algorithms Tested** | 9 (Linear, Polynomial, Lasso, Ridge, ElasticNet, SVR, DT, RF, GBR, XGBoost) |

---

## 🎯 Business Objective

> Help students **understand their realistic admission chances** based on their academic profile — enabling better preparation, goal-setting, and smarter university shortlisting.

**Key Questions Answered:**
- Which academic factors most strongly predict graduate admission?
- Can we accurately predict admission probability from GRE, TOEFL, and GPA?
- Which ML model is most accurate *and* interpretable for this use case?

---

## 📖 Dataset — Feature Dictionary

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `GRE Score` | Continuous | 260–340 | Graduate Record Exam score |
| `TOEFL Score` | Continuous | 0–120 | English proficiency test score |
| `University Rating` | Ordinal | 1–5 | University prestige rating |
| `SOP` | Ordinal | 1.0–5.0 | Statement of Purpose strength |
| `LOR` | Ordinal | 1.0–5.0 | Letter of Recommendation strength |
| `CGPA` | Continuous | 0–10 | Undergraduate GPA |
| `Research` | Binary | 0 / 1 | Research experience (0=No, 1=Yes) |
| `Chance of Admit` | **Target** | 0.0–1.0 | Admission probability |

---

## 🗺️ Project Workflow

```
Load & Explore Data (500 records × 8 features)
              ↓
Data Cleaning
  → Drop Serial No. (unique ID, no predictive value)
  → Fix column name trailing spaces
  → Verify: no missing values, no duplicates, no outliers
              ↓
EDA
  → Correlation heatmap (CGPA, GRE, TOEFL = top predictors)
  → Distribution + scatter plots for top 3 features
  → Pairplot
  → Research experience impact analysis
              ↓
Feature Scaling → MinMaxScaler [0, 1]
              ↓
Smart Random State Selection (100 iterations → pick most stable split)
              ↓
Model Building — 9 Algorithms with GridSearchCV
              ↓
Compare: R² · CV Score · Test R²
              ↓
Best Model: Linear Regression → Residual analysis + OLS summary
```

---

## 📊 Key EDA Insights

| Insight | Finding |
|---------|---------|
| **Top predictor** | CGPA — highest correlation (~0.88) with admission chance |
| **2nd predictor** | GRE Score (~0.80 correlation) |
| **3rd predictor** | TOEFL Score (~0.79 correlation) |
| **Research impact** | Students with research experience have ~15% higher admission probability |
| **Data shape** | Strong linear relationships → ideal for regression models |

---

## 🤖 Models Compared

| # | Model | Type | Test R² | Notes |
|---|-------|------|---------|-------|
| 1 | **Linear Regression** ⭐ | Linear | **~0.84** | Best balance of accuracy + interpretability |
| 2 | Polynomial Regression (deg=2) | Non-linear | ~0.83 | Slight improvement, more complex |
| 3 | Lasso Regression | Regularised | ~0.80 | L1 penalty — auto feature selection |
| 4 | Ridge Regression | Regularised | ~0.84 | L2 penalty — matches Linear R² |
| 5 | Elastic Net | Regularised | ~0.82 | L1 + L2 combined |
| 6 | SVR | Kernel-based | ~0.82 | Requires StandardScaler |
| 7 | Decision Tree Regressor | Tree | ~0.76 | Overfits without pruning |
| 8 | Random Forest Regressor | Ensemble | ~0.83 | Strong but no gain over LR |
| 9 | Gradient Boosting / XGBoost | Boosting | ~0.83 | High train R², similar test R² |

> 🏆 **Linear Regression selected** — matches complex models in accuracy, trains instantly, and provides interpretable coefficients for each academic factor.

---

## 💡 Why Linear Regression Wins Here

```
Data has strong linear relationships (proven by pairplot + heatmap)
           ↓
Complex models (RF, XGBoost) don't capture additional patterns
           ↓
Ensemble models overfit on 500 records — limited training data
           ↓
Linear Regression: same Test R² + interpretable + 100x faster
           ↓
OLS summary confirms: all 7 features are statistically significant (p < 0.05)
```

---

## 🛠️ Smart Random State Selection

Instead of arbitrarily choosing `random_state=42`, we **tested 100 different random states** and selected the one with the most stable Train/Test R² gap:

```python
for i in range(1, 100):
    x_train, x_test = train_test_split(x, y, test_size=0.2, random_state=i)
    # Track R² stability across splits
    
# → Selected random_state=62 for most consistent results
```

This ensures our evaluation isn't influenced by a lucky or unlucky data split.

---

## 📂 Project Structure

```
Admission_Prediction/
│
├── 📓 Admission_project.ipynb        # Full notebook (EDA → 9 models)
├── 📊 admission_predict.csv          # Dataset (500 student records)
└── 📝 README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core language |
| **Pandas / NumPy** | Data loading and manipulation |
| **Matplotlib / Seaborn** | EDA visualisations |
| **Scikit-learn** | All ML models, scaling, GridSearchCV |
| **Statsmodels** | OLS regression + p-value significance testing |
| **XGBoost** | Extreme Gradient Boosting regressor |

---

## ▶️ Run Locally

```bash
git clone https://github.com/SireeshaRagipati24/Machine-Learning.git
cd Machine-Learning/Regression
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost
jupyter notebook Admission_project.ipynb
```

---

## 💡 Key ML Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| Regression Evaluation | R², MAE, RMSE, residual plots |
| Regularisation | Lasso (L1), Ridge (L2), ElasticNet (L1+L2) |
| Polynomial Features | Degree selection (1–9) to avoid overfitting |
| Feature Importance | Ensemble method → select importance > 0 |
| Smart Split | 100 random states → pick most stable |
| OLS Analysis | p-values → confirm statistical significance |
| Residual Analysis | Random scatter = assumptions met ✅ |

---

## 🚀 Future Improvements

- [ ] Build a **Streamlit app** — student enters scores → gets instant admission probability
- [ ] Add **SHAP values** for feature explainability (which factor affected your score most?)
- [ ] Expand dataset for better ensemble model performance
- [ ] Try **Neural Networks** (MLPRegressor) for non-linear capture

---

## 🙋‍♀️ About Me

**Sireesha Ragipati** — Aspiring Data Analyst passionate about turning data into actionable decisions.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/sireesha-ragipati-269a10244/)

---

*⭐ If you found this helpful, give it a star!*
