# 🎬 Movie Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

> **A Content-Based Movie Recommendation System** that suggests similar movies using **TF-IDF vectorization** and **Cosine Similarity** — complete with typo-tolerant fuzzy search and ranked output.

---

## 🎯 Problem Statement

With thousands of movies available, users struggle to discover films that match their taste. This project builds an **intelligent recommendation engine** that, given a movie title, returns the **Top 10 most similar movies** based on genre profile — without requiring any user history or ratings.

---

## 📊 Demo

```
Enter your favourite movie: iron man

🎬 You searched for: 'iron man'
🔍 Best match found: 'Iron Man'

✨ Top 10 Movies Recommended for You:

Rank   Movie Title                                   Similarity Score
----------------------------------------------------------------------
  1    John Carter                                   1.0000  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  2    Avengers: Age of Ultron                       1.0000  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  3    The Avengers                                  1.0000  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  4    Captain America: Civil War                    1.0000  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  5    Iron Man 3                                    1.0000  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  6    Transformers: Revenge of the Fallen           1.0000  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  7    TRON: Legacy                                  1.0000  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  8    Star Trek Into Darkness                       1.0000  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  9    Pacific Rim                                   1.0000  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  10   Guardians of the Galaxy                       0.9289  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```

---

## 🧠 How It Works

```
User Input (movie name)
        ↓
Fuzzy String Matching (difflib)   ← handles typos
        ↓
Find Movie Index in Dataset
        ↓
Retrieve TF-IDF Feature Vector
        ↓
Compute Cosine Similarity vs all 4,693 movies
        ↓
Sort by Similarity Score (descending)
        ↓
Return Top-N Movie Recommendations
```

### Why TF-IDF + Cosine Similarity?

| Technique | Purpose |
|-----------|---------|
| **TF-IDF** | Converts genre text to numerical vectors; penalizes common genres like "Drama" and rewards rare ones like "Western" |
| **Cosine Similarity** | Measures angle between genre vectors — scale-invariant, works perfectly for text features |
| **difflib fuzzy matching** | Tolerates typos in user input (e.g., `"avngers"` → `"Avengers"`) |

---

## 🗂️ Project Structure

```
Movie_Recommendation_Engine/
│
├── 📓 Movie_RecommendationEngine.ipynb   # Main notebook (EDA + Model + Demo)
├── 📄 movies_recommendation.csv          # Dataset (4,693 TMDB movies)
└── 📝 README.md                          # Project documentation
```

---

## 📦 Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.11 | Core language |
| **Pandas** | Latest | Data loading & manipulation |
| **NumPy** | Latest | Numerical operations |
| **Scikit-learn** | Latest | TF-IDF vectorizer + Cosine Similarity |
| **Difflib** | Built-in | Fuzzy string matching for typo tolerance |

---

## 📋 Dataset

| Feature | Details |
|---------|---------|
| **Source** | TMDB (The Movie Database) |
| **Records** | 4,693 movies |
| **Columns** | `index`, `genres`, `title` |
| **Genre Format** | Space-separated (e.g., `"Action Adventure Sci-Fi"`) |
| **Missing Values** | 27 rows with missing genres → filled with empty string |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/SireeshaRagipati24/Data-Analytics.git
cd Data-Analytics/Movie_Recommendation_Engine
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn jupyter
```

**3. Launch the notebook**
```bash
jupyter notebook Movie_RecommendationEngine.ipynb
```

**4. Run all cells and enter a movie name when prompted!**

---

## 📈 Model Performance Insights

For `"Iron Man"` (Action + Adventure + Sci-Fi):

| Similarity Band | Movie Count | Interpretation |
|----------------|-------------|----------------|
| **1.0** | 52 movies | Identical genre profile |
| **0.9 – 1.0** | ~60 movies | Extremely similar |
| **0.5 – 0.9** | ~200 movies | Highly similar |
| **0.0** | ~2,800 movies | No genre overlap |

---

## ⚖️ Limitations & Future Scope

**Current Limitations:**
- Only uses `genre` — ignores cast, director, plot synopsis
- Genre-only matching can produce "filter bubble" effect

**Planned Improvements:**
- Multi-feature model: add cast, director, keywords, overview
- Collaborative Filtering using user ratings
- Hybrid model (Content + Collaborative)
- Streamlit web app for interactive UI
- BERT embeddings for semantic plot-based similarity

---

## 👩‍💻 Author

**Sireesha Ragipati**  
📧 [LinkedIn](https://www.linkedin.com/in/sireesha-ragipati-269a10244/) 

> *"Building intelligent systems that turn data into decisions."*

---

⭐ **If you found this helpful, please star the repository!**
