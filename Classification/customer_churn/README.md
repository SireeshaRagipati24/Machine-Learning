# 🔮 Customer Churn Prediction
### End-to-End Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-red?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## 📌 Problem Statement

Telecom companies lose millions every year to customer churn. The goal of this project is to **predict which customers are likely to leave** — before they do — so the business can take proactive retention action.

> A **5% reduction in churn** can increase profits by **25–95%** (Harvard Business Review).

---

## 🎯 Objective

Build a machine learning model that:
- Accurately predicts customer churn (Yes/No)
- Identifies the key factors driving churn
- Provides actionable business insights for retention

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | IBM Telco Customer Churn Dataset |
| Rows | 7,043 customers |
| Features | 21 (demographics, services, account info) |
| Target | `Churn` (Yes / No) |
| Class Balance | 73.5% No Churn / 26.5% Churned |

**Key Features:**
- `tenure` — How long the customer has been with the company
- `MonthlyCharges` — Monthly bill amount
- `Contract` — Month-to-month / One year / Two year
- `InternetService` — DSL / Fiber optic / None
- `TotalCharges` — Total amount charged

---

## 🛠️ Tech Stack

```
Language     → Python 3.12
Analysis     → Pandas, NumPy
Visualization → Matplotlib, Seaborn
ML Models    → Scikit-learn, XGBoost
Environment  → Jupyter Notebook
```

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── Customer_Churn_Prediction_Professional.ipynb   ← Main notebook
├── Telco-Customer-Churn.csv                       ← Dataset
└── README.md                                       ← This file
```

---

## 🔄 Project Pipeline

```
Raw Data
   ↓
Data Cleaning
(Fix dtypes, remove nulls, drop duplicates)
   ↓
Exploratory Data Analysis
(Distributions, Correlations, Churn patterns)
   ↓
Feature Engineering
(Encoding, Scaling, Train-Test Split)
   ↓
Model Building × 8
(LR, KNN, SVM, DT, RF, AdaBoost, GB, XGBoost)
   ↓
Hyperparameter Tuning
(GridSearchCV with 5-Fold CV)
   ↓
Model Comparison
(Accuracy, ROC-AUC, CV Score)
   ↓
Best Model → XGBoost ✅
```

---

## 🤖 Models Trained

| Model | Type | Tuned |
|-------|------|-------|
| Logistic Regression | Linear | ✅ |
| K-Nearest Neighbors | Distance-based | ✅ |
| Support Vector Machine | Margin-based | ✅ |
| Decision Tree | Tree-based | ✅ |
| Random Forest | Ensemble (Bagging) | ✅ |
| AdaBoost | Ensemble (Boosting) | ✅ |
| Gradient Boosting | Ensemble (Boosting) | ✅ |
| **XGBoost** | **Ensemble (Boosting)** | ✅ |

---

## 📈 Results

| Model | Test Accuracy | ROC-AUC |
|-------|:------------:|:-------:|
| Logistic Regression | ~80% | ~0.85 |
| K-Nearest Neighbors | ~78% | ~0.83 |
| Support Vector Machine | ~80% | ~0.85 |
| Decision Tree | ~78% | ~0.80 |
| Random Forest | ~80% | ~0.85 |
| AdaBoost | ~80% | ~0.85 |
| Gradient Boosting | ~81% | ~0.86 |
| **XGBoost** ⭐ | **~81%** | **~0.87** |

> **Winner: XGBoost** — Best balance of accuracy and generalization

---

## 🔍 Key Business Insights

1. **📄 Contract Type** — Month-to-month customers churn **3× more** than annual contract customers
2. **⏳ Tenure** — Customers who stay past **12 months** are significantly less likely to leave
3. **💰 Monthly Charges** — Customers paying **>$70/month** have higher churn rates
4. **🌐 Fiber Optic** — Despite faster speeds, fiber customers churn more — service quality issue
5. **🔒 No Security Services** — Customers without Online Security or Tech Support churn significantly more

---

## 💡 Business Recommendations

- 🎯 Offer annual contract upgrades to month-to-month customers with incentives
- 💸 Provide first-year loyalty discounts to reduce early churn
- 🔐 Bundle security + tech support in standard plans to increase stickiness
- 📞 Create dedicated support channels for senior citizens (higher churn group)
- 🚨 Use this model to flag high-risk customers and trigger retention campaigns proactively

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

**3. Launch Jupyter Notebook**
```bash
jupyter notebook Customer_Churn_Prediction_Professional.ipynb
```

**4. Run all cells**
`Kernel → Restart & Run All`

---

## 👩‍💻 Author

**Sireesha Ragipati**
Associate Data Scientist | ML & GenAI Enthusiast
Hyderabad, India

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/sireesha-ragipati-269a10244/)

---

## 📄 License

MIT License — free to use and modify.

---

*If this project helped you, give it a ⭐ on GitHub!*
