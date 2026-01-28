# 📊 Customer Churn Prediction using Machine Learning & ANN

## 📌 Project Overview
Customer churn is a critical business problem where organizations risk losing customers to competitors.  
This project focuses on building an end-to-end **Customer Churn Prediction System** using machine learning and deep learning techniques to identify customers who are likely to discontinue services.  
The solution helps businesses take proactive retention actions and improve customer lifetime value.

---

## 🎯 Problem Statement
Predict whether a customer will churn (leave the service) based on demographic details, service usage patterns, and billing information using historical customer data.

---

## 🧠 Solution Approach
The project follows a complete **Data Science lifecycle**, including:

1. Data Understanding & Exploration  
2. Data Cleaning & Validation  
3. Feature Engineering & Encoding  
4. Data Scaling  
5. Model Building (ANN)  
6. Model Evaluation  
7. Business Insights & Interpretation  

---

## 📂 Dataset Information
- **Dataset:** Telco Customer Churn Dataset (IBM)
- **Records:** 7,000+ customers
- **Target Variable:** `Churn` (Yes / No)

### Key Features:
- Customer demographics (Gender, SeniorCitizen, Partner, Dependents)
- Account information (Tenure, Contract, Payment Method)
- Service usage (Internet, Phone, Streaming, Tech Support)
- Billing details (Monthly Charges, Total Charges)

---

## 🔍 Data Preprocessing
- Verified data types and column integrity
- Identified and handled missing and invalid values
- Converted categorical variables into numerical format
- Applied One-Hot Encoding to multi-category features
- Scaled numerical features using **MinMaxScaler**
- Removed duplicates and ensured data consistency

---

## ⚙️ Feature Engineering
- Binary encoding for Yes/No variables
- One-hot encoding for service and contract features
- Separation of numerical and categorical variables
- Prevented multicollinearity using `drop_first=True`

---

## 🤖 Model Building
- **Model Used:** Artificial Neural Network (ANN)
- **Framework:** TensorFlow / Keras
- **Architecture:**
  - Input Layer
  - Hidden Layers with ReLU activation
  - Output Layer with Sigmoid activation
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam

---

## 📈 Model Evaluation
- Achieved approximately **83% accuracy** on test data
- Evaluated model using:
  - Accuracy
  - Confusion Matrix
  - Precision, Recall, and F1-score
- Focused on recall to reduce false negatives (missing churn customers)

---

## 📊 Key Insights
- Customers with **month-to-month contracts** show higher churn rates
- **Low tenure** customers are more likely to churn
- Higher **monthly charges** increase churn probability
- Long-term contracts and bundled services reduce churn risk

---

## 💼 Business Impact
- Enables businesses to identify high-risk customers in advance
- Supports targeted retention strategies
- Improves customer satisfaction and reduces revenue loss
- Demonstrates practical application of machine learning in real-world business problems

---

## 🛠️ Technologies Used
- **Programming:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn  
- **Machine Learning:** Scikit-learn  
- **Deep Learning:** TensorFlow, Keras  
- **Tools:** Jupyter Notebook, GitHub  

---

## 🚀 Future Enhancements
- Implement class imbalance handling (SMOTE / class weighting)
- Compare ANN performance with Logistic Regression and Tree-based models
- Deploy the model using Flask or FastAPI
- Add customer churn probability scoring dashboard

---

## 👩‍💻 Author
**Sireesha Ragipati**  
📍 Hyderabad, India  
📧 ragipatisireesha.job@gmail.com  
🔗 GitHub: https://github.com/SireeshaRagipati24  
🔗 LinkedIn: https://www.linkedin.com/in/sireesha-ragipati-269a10244 
