# Project 3 – AI Based Fake News Detection Tool

**CodeVedX AI/ML Internship 2026 | Akanksha Sudabattula**

---

## Overview

An NLP-based tool that classifies news as **FAKE** or **REAL** using TF-IDF vectorization and PassiveAggressiveClassifier. Includes confidence scores, batch testing, and data visualizations.

---

## Features

| # | Feature |
|---|---------|
| 1 | Detect single news article as FAKE or REAL |
| 2 | Train / retrain model and save to disk |
| 3 | Batch test 8 news samples at once |
| 4 | Generate 3 analysis charts |

---

## Tech Stack

- **Language** – Python 3.x
- **NLP** – TF-IDF Vectorization (scikit-learn)
- **Model** – PassiveAggressiveClassifier
- **Libraries** – Pandas, NumPy, Scikit-learn, Matplotlib
- **Storage** – Pickle (model saved to disk)

---

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib
python fake_news_detector.py
```

---

## NLP Pipeline

```
Raw Text
   ↓
Preprocessing (lowercase, remove special chars, remove short words)
   ↓
TF-IDF Vectorization (text → numbers, ngram_range=(1,2))
   ↓
PassiveAggressiveClassifier
   ↓
FAKE / REAL + Confidence Score
```

---

## Model Results

| Metric | Score |
|--------|-------|
| Accuracy | 93.75% |
| FAKE Precision | 89% |
| REAL Precision | 100% |
| Batch Test Accuracy | 88% |

---

## File Structure

```
Project-3-Fake-News-Detection/
├── fake_news_detector.py   ← main application
├── news_data.csv           ← training dataset (78 samples)
├── fake_news_model.pkl     ← saved ML model
├── tfidf_vectorizer.pkl    ← saved TF-IDF vectorizer
└── README.md
```

---

*Built as part of CodeVedX AI/ML Virtual Internship – Batch 2026*
