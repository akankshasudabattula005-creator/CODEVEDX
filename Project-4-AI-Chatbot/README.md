# Project 4 – AI Chatbot for Internal Helpdesk (Flask Web Version)

**CodeVedX AI/ML Internship 2026 | Akanksha Sudabattula**

---

## Overview

A full-featured AI helpdesk chatbot with a **Flask web interface** — runs in the browser with a beautiful dark-themed chat UI. Powered by TF-IDF vectorization and cosine similarity for intent detection.

---

## Features

| Feature | Description |
|---------|-------------|
| 🌐 Web Interface | Beautiful dark-themed chat UI in browser |
| 🤖 Intent Detection | 20 intents, 106 patterns, cosine similarity |
| ⚡ Real-time Chat | Typing indicator, animated message bubbles |
| 📊 Live Stats | Message count, confidence score, intent display |
| 🔘 Quick Buttons | 10 pre-built question shortcuts in sidebar |
| 🔧 Admin API | Add FAQs and retrain via REST endpoints |
| 💾 Chat History | Session-based history tracking |

---

## Tech Stack

- **Backend** – Python Flask
- **NLP** – TF-IDF + Cosine Similarity (Scikit-learn)
- **Frontend** – HTML, CSS, JavaScript (vanilla)
- **Storage** – Pickle (model), CSV (FAQ dataset)

---

## How to Run

```bash
pip install flask pandas numpy scikit-learn
python app.py
```

Then open your browser at: **http://127.0.0.1:5000**

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main chat interface |
| `/chat` | POST | Send message, get response |
| `/history` | GET | Get chat history |
| `/intents` | GET | List all intents |
| `/retrain` | POST | Retrain chatbot |
| `/add_faq` | POST | Add new FAQ entry |

---

## File Structure

```
Project-4-AI-Chatbot/
├── app.py              ← Flask backend
├── faq_dataset.csv     ← Training data (20 intents)
├── chatbot_model.pkl   ← Saved model (auto-created)
├── templates/
│   └── index.html      ← Chat web interface
└── README.md
```

---

## NLP Pipeline

```
User Message (browser)
      ↓
Flask /chat endpoint
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity with 106 patterns
      ↓
Best intent + confidence score
      ↓
Response → JSON → Browser UI
```

---

*Built as part of CodeVedX AI/ML Virtual Internship – Batch 2026*
