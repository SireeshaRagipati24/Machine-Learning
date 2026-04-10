
# 📈 Google Stock Price Prediction — RNN with Stacked LSTM

> **Teaching a neural network to read market patterns** — using Deep Learning to predict Google's stock price from 5 years of historical data.

---

## 📌 Project Overview

| Detail | Info |
|--------|------|
| **Domain** | Finance / Time Series Forecasting |
| **Algorithm** | Recurrent Neural Network (RNN) with Stacked LSTM |
| **Framework** | TensorFlow / Keras |
| **Training Data** | Google Stock Price — Jan 2012 to Dec 2016 (1,258 trading days) |
| **Test Data** | January 2017 (20 unseen trading days) |
| **Architecture** | 4 Stacked LSTM Layers × 50 Units + Dropout(0.2) |
| **Task** | Predict next-day opening price from 60 previous days |

---

## 🎯 Business Objective

> Can a neural network **learn the hidden patterns in stock price movement** and predict future prices?

This project builds a deep learning model that reads **60 consecutive days of Google's opening stock price** and predicts the next day's price — mimicking how a trader analyses recent history before making a decision.

---

## 🧠 Why LSTM for Stock Prediction?

Standard neural networks treat each input independently — they have **no memory**.
Stock prices are **sequential** — today's price depends on the last 30–60 days of movement.

| Model | Memory | Limitation | Suitable For |
|-------|--------|------------|-------------|
| ANN | ❌ None | Cannot handle sequences | Tabular, independent data |
| RNN | ⚠️ Short-term | Vanishing gradient problem | Short sequences only |
| **LSTM** | ✅ Long-term | Computationally heavier | **Time series, stock prices, NLP** |

> 🔑 **LSTM (Long Short-Term Memory)** solves the vanishing gradient problem using **gates** (input, forget, output) that control what information to remember and what to discard across long sequences.

---

## 🏗️ Model Architecture

```
Input: (60 timesteps × 1 feature — Opening Price)
              ↓
┌─────────────────────────────────────┐
│  LSTM Layer 1  —  50 units          │  return_sequences = True
│  Dropout 20%                        │  ← prevents overfitting
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  LSTM Layer 2  —  50 units          │  return_sequences = True
│  Dropout 20%                        │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  LSTM Layer 3  —  50 units          │  return_sequences = True
│  Dropout 20%                        │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  LSTM Layer 4  —  50 units          │  return_sequences = False
│  Dropout 20%                        │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Dense Output Layer  —  1 unit      │  → predicted price (USD)
└─────────────────────────────────────┘
```

| Component | Choice | Reason |
|-----------|--------|--------|
| LSTM units | 50 | Balance between capacity and overfitting |
| Stacked layers | 4 | Deeper network learns more complex temporal patterns |
| Dropout | 20% | Randomly deactivates neurons → forces robustness |
| `return_sequences` | True (layers 1–3) | Passes full sequence to next LSTM layer |
| Optimizer | Adam | Adaptive learning rate — ideal for RNNs |
| Loss | MSE | Penalises large prediction errors more heavily |

---

## 🗺️ Project Workflow

```
Load & Explore Data (1,258 training days)
            ↓
Feature Scaling  →  MinMaxScaler [0, 1]
            ↓
Create Sequences  →  60 timesteps → predict 1
            ↓
Reshape to 3D  →  (samples × timesteps × features)
            ↓
Build Stacked LSTM  →  4 layers × 50 units + Dropout
            ↓
Train  →  Adam optimizer, MSE loss, 10 epochs
            ↓
Evaluate  →  RMSE · MAE · MAPE
            ↓
Visualise  →  Real vs Predicted + Error Chart
```

---

## 💡 The Sliding Window Concept

The model doesn't learn from individual prices — it learns from **sequences of 60 days**:

```
Day 1–60   →  Predict Day 61
Day 2–61   →  Predict Day 62
Day 3–62   →  Predict Day 63
    ...
```

**Why 60 timesteps?**
- Too few (e.g. 20) → model misses longer market trends
- Too many (e.g. 120) → old data becomes irrelevant, adds noise
- **60 days (~3 months)** → captures meaningful market cycles — industry standard

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **RMSE (Test Set)** | ~$14–18 |
| **MAE (Test Set)** | ~$12–15 |
| **MAPE** | ~1.8–2.5% |
| **Trend Capture** | ✅ Upward trend correctly predicted |

> The model successfully captures the **upward trend** of Google stock in January 2017 — the predicted line closely tracks the real price direction.

---

## 📂 Project Structure

```
RNN_LSTM_Stock_Prediction/
│
├── 📓 RNN_LSTM_Google_stockPrice_prediction.ipynb   # Full DL notebook
├── 📊 Google_Stock_Price_Train.csv                  # Training data (2012–2016)
├── 📊 Google_Stock_Price_Test.csv                   # Test data (Jan 2017)
├── 🖼️  prediction_vs_real.png                       # Output chart
└── 📝 README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core language |
| **TensorFlow / Keras** | LSTM model building and training |
| **NumPy** | Array operations and 3D reshaping |
| **Pandas** | Data loading and manipulation |
| **Scikit-learn** | MinMaxScaler + RMSE evaluation |
| **Matplotlib** | Visualisation of real vs predicted prices |

---

## ▶️ Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/SireeshaRagipati24/Machine-Learning.git
cd Machine-Learning
```

### 2. Install dependencies
```bash
pip install tensorflow keras numpy pandas scikit-learn matplotlib
```

### 3. Run the notebook
```bash
jupyter notebook RNN_LSTM_Google_stockPrice_prediction.ipynb
```

---

## 💡 Key Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| Time Series Preprocessing | Sliding window (60 timesteps) |
| Feature Scaling | MinMaxScaler → [0,1], fit on train only |
| 3D Tensor Reshaping | `(samples, timesteps, features)` for Keras |
| Vanishing Gradient Fix | LSTM gates (forget, input, output) |
| Overfitting Prevention | Dropout(0.2) after each LSTM layer |
| Stacked LSTM | `return_sequences=True` for layers 1–3 |
| Inverse Transformation | Scale predictions back to real USD |

---

## 🚀 Future Improvements

- [ ] Add more features — Close, High, Low, Volume (**multivariate LSTM**)
- [ ] Increase epochs to 50–100 for better convergence
- [ ] Try **Bidirectional LSTM** for richer temporal learning
- [ ] Add **Early Stopping** callback to avoid overfitting
- [ ] Compare with **GRU** — lighter, faster alternative to LSTM
- [ ] Implement **attention mechanism** for long-range dependencies

---

## 🙋‍♀️ About Me

**Sireesha Ragipati** — Aspiring Data Analyst passionate about turning raw data into meaningful predictions.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/sireesha-ragipati-269a10244/)

---

*⭐ If you found this helpful, give it a star!*
