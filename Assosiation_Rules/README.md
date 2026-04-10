# 🛒 Market Basket Analysis — Association Rule Mining with Apriori

> **Discovering hidden purchasing patterns in customer transactions** — using the Apriori algorithm to find which products are bought together and turning that into actionable retail strategy.

---

## 📌 Project Overview

| Detail | Info |
|--------|------|
| **Domain** | Retail / E-commerce Analytics |
| **Technique** | Association Rule Mining — Apriori Algorithm |
| **Library** | mlxtend |
| **Dataset** | 6 customer transactions × 7 grocery items |
| **Key Discovery** | `{jam} → {bread}` — 100% Confidence, 1.2 Lift |
| **Goal** | Find which products are frequently bought together |

---

## 🎯 Business Objective

> Discover **hidden co-purchasing patterns** in customer transaction data — enabling smarter decisions about product placement, cross-selling, bundle promotions, and inventory management.

**Real-World Applications:**
- 🛒 **Supermarket layout** — place co-purchased items near each other
- 💻 **"Customers also bought"** — Amazon/Flipkart recommendation engine
- 📦 **Bundle promotions** — discount pairs of associated products
- 📊 **Inventory planning** — restock associated items together

---

## 🧠 What is Association Rule Mining?

Association Rule Mining finds **IF → THEN relationships** in transactional data:

```
IF customer buys {jam}   → THEN they also buy {bread}  (100% of the time)
IF customer buys {bread} → THEN they also buy {jam}    (80% of the time)
```

### Three Key Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Support** | P(A ∩ B) | How often A and B appear together across ALL transactions |
| **Confidence** | P(B\|A) = P(A∩B)/P(A) | Given A is bought, how often is B also bought? |
| **Lift** | Confidence / P(B) | How much more likely is B bought WITH A vs randomly? |

```
Lift > 1  →  Positive association (items bought together more than by chance) ✅
Lift = 1  →  No association (independent items)
Lift < 1  →  Negative association (items avoid each other)
```

---

## 🗺️ Project Workflow

```
Define Transaction Data (6 shopping baskets)
               ↓
Visualise Item Frequency & Support
               ↓
One-Hot Encoding → TransactionEncoder (binary matrix)
               ↓
Apriori Algorithm → Find Frequent Itemsets (min_support=0.6)
               ↓
Generate Association Rules (min_confidence=0.6)
               ↓
Visualise: Support vs Confidence vs Lift
               ↓
Interpret Rules (support, confidence, lift, conviction)
               ↓
Business Recommendations
               ↓
Threshold Sensitivity Analysis
```

---

## 📊 Transaction Data

| Transaction | Items |
|-------------|-------|
| T1 | milk, bread, rice, book |
| T2 | bread, jam, book, pen |
| T3 | jam, milk, bread, rice, eggs |
| T4 | rice, eggs, pen, book |
| T5 | eggs, pen, milk, bread, jam |
| T6 | eggs, rice, bread, jam |

**Item support values:**
```
bread  ████████████████  0.833  ✅ frequent
eggs   █████████████     0.667  ✅ frequent
jam    █████████████     0.667  ✅ frequent
rice   █████████████     0.667  ✅ frequent
book   ██████████        0.500  ❌ below threshold
milk   ██████████        0.500  ❌ below threshold
pen    ██████████        0.500  ❌ below threshold
```

---

## 🔍 Results — Association Rules Found

| Rule | Support | Confidence | Lift | Verdict |
|------|---------|------------|------|---------|
| `{jam} → {bread}` | 0.667 (66.7%) | **1.0 (100%)** | 1.2 | ⭐ Perfect rule |
| `{bread} → {jam}` | 0.667 (66.7%) | 0.8 (80%) | 1.2 | ✅ Strong rule |

### Rule 1: `{jam} → {bread}` — The Strongest Rule

- **Support 66.7%:** 4 out of 6 transactions contain both jam and bread
- **Confidence 100%:** Every single customer who bought jam also bought bread
- **Lift 1.2:** Buying bread is 20% more likely when jam is in the cart
- **Conviction ∞:** Perfect deterministic rule — zero exceptions

### Rule 2: `{bread} → {jam}` — Strong but Asymmetric

- **Confidence 80%:** 4 out of 5 bread buyers also bought jam
- **Asymmetry:** The relationship is NOT symmetric — jam is a stronger predictor of bread than vice versa

---

## 💡 Business Recommendations

| Recommendation | Based On |
|---------------|----------|
| 🛒 **Place jam next to bread** on shelves | 100% of jam buyers also buy bread |
| 💰 **Create "Breakfast Bundle"** (jam + bread discount) | 66.7% support — very common pair |
| 📱 **Recommend bread when jam added to cart** | 100% confidence rule |
| 📦 **Sync jam & bread restocking** | Always purchased together |

---

## 📂 Project Structure

```
Association_Rules/
│
├── 📓 Assosiation_rules.ipynb    # Full notebook (end-to-end)
└── 📝 README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core language |
| **mlxtend** | TransactionEncoder, Apriori, association_rules |
| **Pandas / NumPy** | Data manipulation |
| **Matplotlib / Seaborn** | Visualisations |

---

## ▶️ Run Locally

```bash
git clone https://github.com/SireeshaRagipati24/Machine-Learning.git
cd Machine-Learning/Association_Rules

pip install mlxtend pandas numpy matplotlib seaborn

jupyter notebook Assosiation_rules.ipynb
```

---

## 💡 Key Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| Transaction Encoding | TransactionEncoder → binary True/False matrix |
| Apriori Algorithm | Bottom-up frequent itemset generation |
| Support Filtering | min_support=0.6 → 4+ of 6 transactions |
| Confidence Filtering | min_confidence=0.6 → rule holds 60%+ of time |
| Lift Interpretation | >1 = positive, =1 = independent, <1 = negative |
| Rule Asymmetry | jam→bread ≠ bread→jam (directional relationship) |
| Threshold Sensitivity | Impact of varying support and confidence cutoffs |

---

## 🔄 How Apriori Works (Step by Step)

```
Step 1 — Frequent Single Items (support ≥ 0.6):
  bread(0.83), eggs(0.67), jam(0.67), rice(0.67) ✅
  book(0.50), milk(0.50), pen(0.50)              ❌

Step 2 — Frequent Item Pairs (support ≥ 0.6):
  {jam, bread}(0.67) ✅
  All other pairs fall below threshold            ❌

Step 3 — Frequent Triplets:
  None meet threshold                             ❌

→ Generate rules from frequent pair {jam, bread}:
  {jam} → {bread}  confidence = 0.67/0.67 = 1.0  ✅
  {bread} → {jam}  confidence = 0.67/0.83 = 0.8  ✅
```

---

## 🚀 Future Improvements

- [ ] Apply to real dataset — **UCI Online Retail Dataset** (500K+ transactions)
- [ ] Try **FP-Growth algorithm** — faster alternative for large-scale data
- [ ] Build **network graph** — visualise all item relationships as nodes/edges
- [ ] Add **seasonal analysis** — different rules for weekdays vs weekends
- [ ] Deploy as a **product recommendation API**

---

## 🙋‍♀️ About Me

**Sireesha Ragipati** — Aspiring Data Analyst passionate about turning raw data into meaningful business decisions.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/sireesha-ragipati-269a10244/)

---

*⭐ If you found this helpful, give it a star!*
