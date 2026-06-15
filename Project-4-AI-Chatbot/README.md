# Project 4 – AI Chatbot for Internal Helpdesk

**CodeVedX AI/ML Internship 2026 | Akanksha Sudabattula**

---

## Overview

An NLP-based helpdesk chatbot that detects user intent and responds to internship-related queries. Trained on a custom FAQ dataset with 20 intents and 106 patterns.

---

## Features

| # | Feature |
|---|---------|
| 1 | Intent-based chat with confidence scores |
| 2 | 20 intents covering all internship queries |
| 3 | Admin panel to add new FAQs |
| 4 | Retrain chatbot without restarting |
| 5 | Chat history saved to JSON |
| 6 | Fallback responses for unknown queries |

---

## Tech Stack

- **Language** – Python 3.x
- **NLP** – TF-IDF Vectorization + Cosine Similarity
- **Libraries** – Pandas, NumPy, Scikit-learn
- **Storage** – Pickle (model), JSON (chat history), CSV (FAQ dataset)

---

## How to Run

```bash
pip install pandas numpy scikit-learn
python chatbot.py
```

---

## Intent Detection Pipeline

```
User Input
   ↓
Preprocessing (lowercase, remove special chars)
   ↓
TF-IDF Vectorization (ngram_range=(1,2))
   ↓
Cosine Similarity with all training patterns
   ↓
Best matching Intent + Confidence Score
   ↓
Response from FAQ Dataset
```

---

## Test Results

**20/20 intents detected correctly — 100% accuracy**

---

## Supported Intents

greeting, farewell, thanks, course_info, internship_info, certificate, fees, duration, contact, github, linkedin, submission, mentor, telegram, deadline, plagiarism, stipend, skills, project_count, recommendation

---

## File Structure

```
Project-4-AI-Chatbot/
├── chatbot.py          ← main application
├── faq_dataset.csv     ← training data (20 intents, 106 patterns)
├── chatbot_model.pkl   ← saved model
├── chat_history.json   ← auto-saved chat logs
└── README.md
```

---

*Built as part of CodeVedX AI/ML Virtual Internship – Batch 2026*
