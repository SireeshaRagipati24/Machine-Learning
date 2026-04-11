# 🍽️ Predictive Restaurant Recommender

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Geospatial-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

> **A location-aware restaurant recommendation engine** for a food delivery platform — combining **Haversine GPS distance** with **historical order behavior scoring** to recommend the most relevant vendors for each customer.

---

## 🎯 Problem Statement

A food delivery company wants to predict which restaurants each customer is most likely to order from. Given a customer's GPS location and past order behavior across the platform, recommend the **Top-5 most relevant vendors** for every customer-location pair in the test set.

---

## 🧠 Solution Architecture

```
Train Data (4 tables)
       ↓
Data Cleaning & Feature Engineering
       ↓
Vendor Quality Score (from order history)
       ↓
Haversine Distance (customer ↔ vendor GPS)
       ↓
Hybrid Scoring = 60% Proximity + 40% Vendor Quality
       ↓
Top-5 Recommendations per Customer-Location Pair
```

---

## 📊 Dataset Overview

| Table | Rows | Key Features |
|-------|------|--------------|
| `train_customers` | 34,674 | customer_id |
| `train_locations` | 59,503 | customer_id, lat/lon |
| `train_orders` | 135,303 | customer_id, vendor_id, ratings, favorites |
| `vendors` | 100 | vendor_id, lat/lon, cuisine tags, ratings |
| `test_customers` | 9,768 | customer_id |
| `test_locations` | 16,720 | customer_id, lat/lon |

**Source:** Soulpage IT Solutions Food Delivery Platform (Kaggle Competition)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.11** | Core language |
| **Pandas** | Data loading, cleaning, merging |
| **NumPy** | Vectorized Haversine calculations |
| **Scikit-learn** | Feature normalization |

---

## 🗂️ Project Structure

```
Restaurant_Recommender/
│
├── 📓 Sireesha_Restaurant_Recommender.ipynb   # Full pipeline notebook
├── 📁 Train_soulpage/                          # Training data folder
│   ├── train_customers.csv
│   ├── train_locations.csv
│   ├── orders.csv
│   └── vendors.csv
├── 📁 Test_soulpage/                           # Test data folder
│   ├── test_customers.csv
│   └── test_locations.csv
├── 📄 submission_final.csv                     # Generated predictions
└── 📝 README.md
```

---

## 🔧 Data Cleaning Summary

### train_customers (34,674 → 34,523 rows)
| Column | Decision | Reason |
|--------|----------|--------|
| `gender` | ❌ Dropped | 35% missing, 93% Male — too imbalanced |
| `dob` | ❌ Dropped | 91% missing |
| `status`, `verified`, `language` | ❌ Dropped | No variation or predictive value |
| `created_at`, `updated_at` | ❌ Dropped | Not predictive for food preference |

### train_orders (135,303 → 135,221 rows)
| Column | Decision | Reason |
|--------|----------|--------|
| `vendor_rating` | ✅ Kept + flag | 66% missing; added `has_vendor_rating` binary flag |
| `is_favorite`, `is_rated` | ✅ Kept + encoded | Strong preference signals → encoded 0/1 |
| `item_count` | ✅ Filled median | Only 5% missing |
| 10 timestamp columns | ❌ Dropped | 37–96% missing; sparse |
| `promo_code` | ❌ Dropped | 97% missing |

### vendors (100 → 97 rows)
- Dropped 28 day-of-week schedule columns (replaced by `is_open`)
- Dropped operational/internal fields
- One-hot encoded `vendor_category_en`

---

## 📐 Haversine Distance Formula

```python
def haversine_single(lat1_rad, lon1_rad, lat2_arr, lon2_arr):
    R = 6371  # Earth radius in km
    dlat = lat2_arr - lat1_rad
    dlon = lon2_arr - lon1_rad
    a = sin(dlat/2)² + cos(lat1_rad) × cos(lat2_arr) × sin(dlon/2)²
    return 2R × arcsin(√a)
```

> **Why Haversine over Euclidean?** GPS coordinates lie on a sphere. Euclidean distance ignores Earth's curvature and gives incorrect results for large distances.

---

## 🏆 Vendor Quality Score

Built from historical order data:

```python
vendor_score = (
    0.35 × order_frequency_norm +    # How popular is this vendor?
    0.30 × avg_rating_norm +          # How well-rated?
    0.20 × favorite_rate_norm +       # How often favorited?
    0.10 × rating_engagement_norm +   # How engaged are customers?
    0.05 × avg_discount_norm          # Does it offer discounts?
)
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/SireeshaRagipati24/Data-Analytics.git
cd Data-Analytics/Restaurant_Recommender
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn jupyter
```

**3. Set up data folders**
```
Place Train_soulpage/ and Test_soulpage/ folders in the same directory as the notebook.
```

**4. Run all cells**
```bash
jupyter notebook Sireesha_Restaurant_Recommender.ipynb
```

Output: `submission_final.csv` with Top-5 vendor recommendations per customer-location.

---

## 📈 Sample Output

```
Customer: LRX7BCH | Location: (0.1122, -78.6042)

Rank  Vendor ID    Distance (km)    Score
----  ---------    -------------    -----
  1   Vendor 82        7.68 km     0.8124
  2   Vendor 157       7.78 km     0.7983
  3   Vendor 310       9.86 km     0.7561
  4   Vendor 265       9.93 km     0.7448
  5   Vendor 160      11.20 km     0.7210
```

---

## 🔮 Limitations & Future Improvements

| Limitation | Planned Enhancement |
|------------|---------------------|
| Location-only model | Add cuisine preference from `vendor_tag_name` |
| No user-user similarity | Collaborative filtering on order history |
| Invalid GPS outliers detected | Add coordinate validation layer |
| Static scoring | Time-aware model (recency, time-of-day) |
| No A/B testing | Track recommendation click-through rate |

---

## 👩‍💻 Author

**Sireesha Ragipati**  
📧 [LinkedIn](https://www.linkedin.com/in/sireesha-ragipati-269a10244/) 

> *"Every customer has a location. Every location has a perfect restaurant nearby."*

---

⭐ **If you found this helpful, please star the repository!**
