# 🏦 Loan Eligibility Prediction System

## 📌 Project Overview
Dream Housing Finance Company provides home loans across urban, semi-urban, and rural areas.  
The objective of this project is to **automate the loan eligibility process in real time** using machine learning based on customer details submitted through an online application.

This system predicts whether a loan application should be **Approved (Y)** or **Rejected (N)** by analyzing applicant demographics, income, credit history, and property details.

---

## 🎯 Business Problem
Manual loan approval is time-consuming and prone to bias.  
The company wants an **automated, data-driven solution** to:
- Quickly assess customer eligibility
- Reduce manual effort
- Target eligible customers effectively
- Improve decision consistency

---

## 📊 Dataset Description
The dataset contains **614 loan applications** with the following features:

| Feature | Description |
|------|------------|
| Gender | Male / Female |
| Married | Applicant marital status |
| Dependents | Number of dependents |
| Education | Graduate / Not Graduate |
| Self_Employed | Employment type |
| ApplicantIncome | Applicant income |
| CoapplicantIncome | Co-applicant income |
| LoanAmount | Loan amount (in thousands) |
| Loan_Amount_Term | Loan term (months) |
| Credit_History | Credit history status |
| Property_Area | Urban / Semiurban / Rural |
| Loan_Status | Target variable (Y / N) |

---

## 🔍 Exploratory Data Analysis (EDA)
- Analyzed **income and loan amount distributions**
- Identified **skewness and outliers**
- Studied the impact of categorical variables on loan approval
- Key insights:
  - Applicants with **good credit history** have a very high approval rate
  - **Semiurban and Urban** applicants show higher approval probability
  - Total income plays a significant role in approval decisions

---

## 🧹 Data Preprocessing
### ✔ Data Cleaning
- Handled missing values using **mode** (categorical) and **row removal** for critical fields
- Converted incorrect data types
- Treated outliers carefully (high-income values retained)

### ✔ Feature Engineering
- Created **Total_income = ApplicantIncome + CoapplicantIncome**
- Dropped irrelevant columns (Loan_ID)

### ✔ Encoding
- Converted categorical variables into numerical format
- Applied label encoding for binary features

### ✔ Transformation
- Applied **Box-Cox transformation** to reduce skewness in:
  - Total_income
  - LoanAmount

---

## 🤖 Machine Learning Models Implemented
Multiple algorithms were trained and evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost

Each model was tuned using **GridSearchCV** and evaluated using:
- Accuracy
- Cross-validation score
- Confusion Matrix
- Classification Report
- ROC-AUC Score

---

## 🏆 Best Model: Decision Tree Classifier
After comparing all models, **Decision Tree** performed best with strong generalization.

### 📈 Performance Metrics
- **Train Accuracy:** ~81%
- **Cross-Validation Accuracy:** ~80%
- **Test Accuracy:** ~84%
- **ROC-AUC Score:** ~0.73

### 🔑 Important Features Identified
- Credit_History (most influential)
- Total_income
- LoanAmount
- Loan_Amount_Term
- Property_Area

---

## 🔮 Prediction on New Data
- New user data is preprocessed using the same pipeline
- Model predicts loan approval in real time
- Output:
  - `1` → Loan Approved
  - `0` → Loan Rejected

---

## 🛠 Technologies Used
- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn**
- **XGBoost**
- **Joblib (Model Saving)**

---

## 🚀 Key Learnings
- Importance of **EDA before modeling**
- Handling **missing values and skewed data**
- Feature engineering significantly improves model performance
- Credit history is the most critical factor in loan approval
- Model comparison is essential before final selection

---

## 🔮 Future Improvements
- Handle class imbalance using SMOTE
- Deploy model using Flask or FastAPI
- Add probability-based approval decision
- Integrate real-time web application

---

## 📌 Conclusion
This project demonstrates an **end-to-end machine learning pipeline**, from business understanding and EDA to model deployment readiness.  
The final model provides a **reliable and scalable solution** for automating loan eligibility decisions, helping financial institutions improve efficiency and customer experience.

---

⭐ *If you find this project useful, feel free to star the repository!*

