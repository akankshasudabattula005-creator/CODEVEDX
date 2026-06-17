# ============================================================
#  AI Chatbot for Internal Helpdesk — Flask Web Version
#  Author : Akanksha Sudabattula
#  Internship : CodeVedX – AI/ML Batch 2026
#  Project 4  : NLP + Intent Detection + Flask Web Interface
# ============================================================

from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import pickle
import os
import re
import json
import random
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = "codevedx_chatbot_2026"

DATA_FILE  = "faq_dataset.csv"
MODEL_FILE = "chatbot_model.pkl"

# ============================================================
# SECTION 1 — NLP Preprocessing
# ============================================================

def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# SECTION 2 — Train / Load Chatbot
# ============================================================

def load_faq_data():
    df   = pd.read_csv(DATA_FILE)
    rows = []
    for _, row in df.iterrows():
        patterns = [p.strip() for p in row["patterns"].split(",")]
        for pattern in patterns:
            rows.append({
                "intent"  : row["intent"],
                "pattern" : preprocess(pattern),
                "response": row["response"]
            })
    return pd.DataFrame(rows), df


def train_chatbot():
    training_data, raw_df = load_faq_data()
    tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words="english", max_features=3000)
    X     = tfidf.fit_transform(training_data["pattern"])

    model_data = {
        "tfidf"        : tfidf,
        "X"            : X,
        "training_data": training_data,
        "raw_df"       : raw_df
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_data, f)
    return model_data


def load_model():
    if not os.path.exists(MODEL_FILE):
        return train_chatbot()
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


# ============================================================
# SECTION 3 — Intent Detection
# ============================================================

def get_fallback():
    fallbacks = [
        "I'm not sure about that. Try asking about courses, internship, certificates, or submissions.",
        "That's outside my knowledge. Please contact contact@codevedx.in for help.",
        "I couldn't find an answer. Reach the team at teamleader@codevedx.in.",
        "Could you rephrase? I can help with internship, GitHub, LinkedIn, certificates, and more.",
    ]
    return random.choice(fallbacks)


def detect_intent(user_input: str, model_data: dict) -> dict:
    tfidf         = model_data["tfidf"]
    X             = model_data["X"]
    training_data = model_data["training_data"]

    clean = preprocess(user_input)
    if not clean:
        return {"intent": "unknown", "confidence": 0, "response": get_fallback()}

    user_vec     = tfidf.transform([clean])
    similarities = cosine_similarity(user_vec, X).flatten()
    best_idx     = np.argmax(similarities)
    best_score   = similarities[best_idx]

    if best_score < 0.15:
        return {"intent": "unknown", "confidence": round(best_score*100,1), "response": get_fallback()}

    best_row   = training_data.iloc[best_idx]
    confidence = min(99.5, best_score * 100 * 1.5)

    return {
        "intent"    : best_row["intent"],
        "confidence": round(confidence, 1),
        "response"  : best_row["response"]
    }


# ============================================================
# SECTION 4 — Flask Routes
# ============================================================

# Load model once at startup
model_data = load_model()


@app.route("/")
def index():
    """Serve the main chat page."""
    session.clear()
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat messages from frontend."""
    data       = request.get_json()
    user_msg   = data.get("message", "").strip()

    if not user_msg:
        return jsonify({"response": "Please type a message.", "intent": "none", "confidence": 0})

    result = detect_intent(user_msg, model_data)

    # Save to session history
    if "history" not in session:
        session["history"] = []

    session["history"].append({
        "time"      : datetime.now().strftime("%H:%M"),
        "user"      : user_msg,
        "bot"       : result["response"],
        "intent"    : result["intent"],
        "confidence": result["confidence"]
    })
    session.modified = True

    return jsonify({
        "response"  : result["response"],
        "intent"    : result["intent"],
        "confidence": result["confidence"]
    })


@app.route("/history")
def history():
    """Return chat history."""
    return jsonify(session.get("history", []))


@app.route("/intents")
def intents():
    """Return all available intents."""
    df      = pd.read_csv(DATA_FILE)
    intents = df["intent"].tolist()
    return jsonify({"intents": intents, "count": len(intents)})


@app.route("/retrain", methods=["POST"])
def retrain():
    """Admin endpoint to retrain the chatbot."""
    global model_data
    model_data = train_chatbot()
    df = pd.read_csv(DATA_FILE)
    return jsonify({
        "status" : "success",
        "message": f"Chatbot retrained on {len(df)} intents successfully!"
    })


@app.route("/add_faq", methods=["POST"])
def add_faq():
    """Admin endpoint to add new FAQ."""
    data     = request.get_json()
    intent   = data.get("intent", "").strip()
    patterns = data.get("patterns", "").strip()
    response = data.get("response", "").strip()

    if not all([intent, patterns, response]):
        return jsonify({"status": "error", "message": "All fields required."})

    df      = pd.read_csv(DATA_FILE)
    new_row = pd.DataFrame([{"intent": intent, "patterns": patterns, "response": response}])
    df      = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

    return jsonify({"status": "success", "message": f"FAQ added for intent: {intent}"})


if __name__ == "__main__":
    print("\n  🤖 CodeVedX AI Helpdesk Chatbot")
    print("  Open your browser at: http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop\n")
    app.run(debug=True, port=5000)
