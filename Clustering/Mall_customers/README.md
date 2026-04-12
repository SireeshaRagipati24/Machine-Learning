
# 🛍️ Mall Customer Segmentation — K-Means vs Hierarchical vs DBSCAN

> **Discovering hidden customer groups** using three unsupervised clustering algorithms on Mall Customer data — and turning those groups into actionable marketing strategies.

---

## 📌 Project Overview

| Detail | Info |
|--------|------|
| **Domain** | Retail / Marketing Analytics |
| **Technique** | Unsupervised Learning — Clustering |
| **Dataset** | Mall Customers — 200 records |
| **Features Used** | Annual Income (k$) · Spending Score (1-100) |
| **Algorithms** | K-Means · Hierarchical (Agglomerative) · DBSCAN |
| **Optimal Clusters** | 5 (confirmed by Elbow + Silhouette + Dendrogram) |

---

## 🎯 Business Objective

> **Segment mall customers into distinct groups** based on annual income and spending behaviour — enabling the marketing team to design **targeted, personalised campaigns** for each customer type.

**Real-World Impact:**
- 🎯 **Targeted marketing** — different promotions for different segments
- 💰 **Revenue optimisation** — focus high-value efforts on high-spending customers
- 🛍️ **Product placement** — stock premium products near high-income zones
- 📊 **Customer lifetime value** — identify and retain VIP customers

---

## 🧠 Clustering vs Classification

| | Classification (Supervised) | Clustering (Unsupervised) |
|--|------|------|
| **Labels** | Given — we know the answer | Unknown — we discover groups |
| **Goal** | Predict a known category | Find hidden structure |
| **Example** | spam / not spam | Which customers are similar? |

---

## 📊 Three Algorithms — When to Use Which

| Algorithm | How It Works | Strengths | Weakness |
|-----------|-------------|-----------|----------|
| **K-Means** | Assign points to nearest centroid, iterate | Fast, scalable | Must specify K; only spherical clusters |
| **Hierarchical** | Merge closest clusters bottom-up | Dendrogram shows K visually | Slow on large data |
| **DBSCAN** | Expand clusters from dense core points | Any shape; detects outliers | Varying density issues |

---

## 🗺️ Project Workflow

```
Load Mall Customers Dataset (200 records)
              ↓
EDA — Distributions, Correlations, Gender split
              ↓
Feature Selection + StandardScaler
              ↓
K-Means → Elbow Method + Silhouette → K=5
              ↓
Hierarchical → Dendrogram → K=5 confirmed
              ↓
DBSCAN → k-NN distance plot → eps tuning
              ↓
All 3 Algorithms Compared Side-by-Side
              ↓
5 Customer Archetypes → Business Strategy
```

---

## 🔍 How We Selected Optimal Parameters

### Elbow Method (K-Means)
```python
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init='k-means++')
    wcss.append(km.inertia_)
# Elbow at K=5 ✅
```

### Dendrogram (Hierarchical)
```
Cut the longest vertical line with no horizontal intersection
→ Number of clusters below cut = optimal K
→ Confirmed K=5 ✅
```

### k-NN Distance Plot (DBSCAN)
```
1. Find k-th nearest neighbour distance for each point
2. Sort descending and plot
3. Knee point = optimal eps value → eps ≈ 0.5 ✅
```

### Silhouette Score (All algorithms)
```
Range: -1 to +1
  +1 = perfect cluster separation
   0 = overlapping clusters
  -1 = misassigned points
```

---

## 💡 5 Customer Segments — Business Strategy

| Segment | Income | Spending | Strategy |
|---------|--------|----------|----------|
| 🌟 **VIP Targets** | High | High | Loyalty programs, premium products, exclusive events |
| 💼 **Potential Converters** | High | Low | Personalised incentives, luxury trials, invite-only sales |
| 🛍️ **Enthusiastic Shoppers** | Low | High | Budget deals, EMI, flash sales, discount clubs |
| 💤 **Low Engagement** | Low | Low | Economy offers only — minimal marketing spend |
| 📊 **Middle Ground** | Medium | Medium | Seasonal campaigns, bundle offers |

> 🎯 **Highest ROI:** High Income + Low Spending — they have money but aren't spending. A personalised incentive could unlock major revenue.

---

## 📂 Project Structure

```
Clustering/
│
├── 📓 Mall_Customer_Segmentation.ipynb  # All 3 algorithms in one notebook
├── 📊 Mall_Customers.csv                # Raw dataset (200 customers)
└── 📝 README.md
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/SireeshaRagipati24/Machine-Learning.git
cd Machine-Learning/Clustering
pip install pandas numpy matplotlib seaborn scikit-learn scipy
jupyter notebook Mall_Customer_Segmentation.ipynb
```

---

## 💡 Key Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| Elbow Method | WCSS vs K — find optimal cluster count |
| Silhouette Score | Universal cluster quality metric |
| Dendrogram | Visual K selection for Hierarchical |
| k-NN Distance Plot | Principled eps selection for DBSCAN |
| StandardScaler | Mandatory preprocessing for distance algorithms |
| Noise Detection | DBSCAN label=-1 → automatic outlier identification |
| Business Translation | Cluster → named segment → marketing strategy |

---

## 🚀 Future Improvements

- [ ] Include Age as a 3rd dimension — 3D clustering
- [ ] Try **HDBSCAN** — handles varying density better
- [ ] Use **PCA** for dimensionality reduction with more features
- [ ] Deploy as **Streamlit app** — input new customer → get segment label
- [ ] Apply **RFM Analysis** for richer segmentation

---

## 🙋‍♀️ About Me

**Sireesha Ragipati** — Aspiring Data Analyst | Unsupervised Learning Enthusiast

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/sireesha-ragipati-269a10244/)

---

*⭐ If you found this helpful, give it a star!*
