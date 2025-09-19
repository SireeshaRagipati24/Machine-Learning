
# Credit Risk Modelling using Machine Learning

**Credit Risk Modelling** is a predictive analytics project designed to **assess the creditworthiness of loan applicants** based on their financial profiles. The project uses **machine learning algorithms** to classify applicants into `Good` or `Bad` credit risk categories.

---

## **Project Objective**
- Predict whether a loan applicant is likely to be a **Good** or **Bad** credit risk.
- Help financial institutions make informed lending decisions.
- Explore data, visualize patterns, and apply machine learning for prediction.

---

## **Dataset**
- **Dataset Used:** https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk
- **Features:**
  - Numerical: `Age`, `Credit amount`, `Duration`
  - Categorical: `Sex`, `Job`, `Housing`, `Saving accounts`, `Checking account`, `Purpose`
- **Target:** `Risk` (`Good` or `Bad`)

---

## **Data Exploration**
- Checked **missing values** and handled them by dropping rows with NaN.
- Analyzed **numerical features** using histograms and boxplots.
- Explored **categorical distributions** with countplots.
- Studied **relationships between features** using correlation and scatterplots.
- Performed **Risk-based visualizations** to understand patterns.

---

## **Data Preprocessing**
- **Label Encoding** for categorical variables (`Sex`, `Housing`, `Saving accounts`, `Checking account`, `Purpose`, `Risk`).
- Saved **label encoders** for deployment using `joblib`.

---

## **Machine Learning Models**
Trained and tuned multiple models using **GridSearchCV**:

| Model                   | Best Accuracy | Key Parameters |
|--------------------------|---------------|----------------|
| Decision Tree Classifier | 58.1%         | max_depth=5, min_samples_split=2, min_samples_leaf=1 |
| Random Forest Classifier | 61.9%         | n_estimators=100, max_depth=None, min_samples_split=10, min_samples_leaf=2 |
| Extra Trees Classifier   | 64.8%         | n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2 |
| XGBoost Classifier       | 66.7%         | n_estimators=100, max_depth=3, learning_rate=0.2, subsample=1, colsample_bytree=0.7 |

**Best Model:** `Extra Trees Classifier` (saved as `extra_tress_credit_model.pkl`)  

---

## **Website / Web App**
- Built a **fully interactive website** using **React + Tailwind CSS + Streamlit / ML API integration**.
- Users can input applicant details:
  - Age, Sex, Job Level, Housing, Savings, Checking Account, Credit Amount, Duration
- **Real-time prediction** of Credit Risk as `GOOD` or `BAD`.
- Clean **UI/UX design**, interactive forms, tooltips, and visual feedback.
- Deployed on **[Your Hosting Platform / URL]** for public access.

---

## **Libraries & Technologies**
- **Python:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`
- **Visualization:** `matplotlib`, `seaborn`
- **Web App / Frontend:** `React`, `Tailwind CSS`
- **Deployment / API:** `Streamlit`, `React Router`, `Axios / Fetch`
- **Version Control:** GitHub

---

## **Results**
- Successfully classified credit risk with **Extra Trees Classifier achieving ~64.8% accuracy**.
- Visualizations highlight how financial attributes like **savings, housing, and job** impact credit risk.
- Web app provides **easy-to-use interface** for financial analysts and institutions.

---

## **Future Improvements**
- Handle **class imbalance** with SMOTE or weighting.
- Feature engineering on **credit ratios, age groups, job types**.
- Test **more advanced ensemble models** or deep learning for higher accuracy.
- Enhance website with **charts for risk analysis** and **multi-user support**.
- Deploy on **cloud (Heroku, AWS, Streamlit Cloud)** for real-time usage.

---

## **How to Run**
1. Clone the repository:  
```bash
git clone <repository-url>
