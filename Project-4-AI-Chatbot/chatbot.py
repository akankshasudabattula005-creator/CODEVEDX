# ============================================================
#  AI Chatbot for Internal Helpdesk
#  Author : Akanksha Sudabattula
#  Internship : CodeVedX – AI/ML Batch 2026
#  Project 4  : NLP + Intent Detection + FAQ Training
# ============================================================

import pandas as pd
import numpy as np
import re
import json
import os
import pickle
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# ── File Paths ────────────────────────────────────────────────
DATA_FILE    = "faq_dataset.csv"
MODEL_FILE   = "chatbot_model.pkl"
HISTORY_FILE = "chat_history.json"


# ============================================================
# SECTION 1 – Text Preprocessing
# ============================================================

def preprocess(text: str) -> str:
    """Normalize text for intent matching."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# SECTION 2 – Load and Train from FAQ Dataset
# ============================================================

def load_faq_data():
    """Load FAQ dataset and expand patterns into training rows."""
    df = pd.read_csv(DATA_FILE)
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
    """
    Train the chatbot using TF-IDF vectorization.
    Each pattern gets a TF-IDF vector.
    At prediction time, we find the most similar pattern.
    """
    training_data, raw_df = load_faq_data()

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=3000
    )

    X = tfidf.fit_transform(training_data["pattern"])

    # Save model
    model_data = {
        "tfidf"        : tfidf,
        "X"            : X,
        "training_data": training_data,
        "raw_df"       : raw_df
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_data, f)

    print(f"  ✔ Chatbot trained on {len(training_data)} patterns across {raw_df['intent'].nunique()} intents.")
    return model_data


def load_chatbot():
    """Load trained chatbot model from disk."""
    if not os.path.exists(MODEL_FILE):
        print("  [INFO] Training chatbot for first time...")
        return train_chatbot()
    with open(MODEL_FILE, "rb") as f:
        model_data = pickle.load(f)
    print(f"  ✔ Chatbot loaded. {len(model_data['raw_df'])} intents ready.")
    return model_data


# ============================================================
# SECTION 3 – Intent Detection
# ============================================================

def detect_intent(user_input: str, model_data: dict) -> dict:
    """
    Detect the intent of user input using cosine similarity.

    Steps:
    1. Preprocess user input
    2. Transform to TF-IDF vector
    3. Compute cosine similarity with all training patterns
    4. Return best matching intent and confidence
    """
    tfidf         = model_data["tfidf"]
    X             = model_data["X"]
    training_data = model_data["training_data"]

    clean_input = preprocess(user_input)

    # Handle empty input
    if not clean_input:
        return {"intent": "unknown", "confidence": 0.0, "response": get_fallback()}

    # Vectorize user input
    user_vector = tfidf.transform([clean_input])

    # Cosine similarity with all patterns
    similarities = cosine_similarity(user_vector, X).flatten()

    # Get best match
    best_idx   = np.argmax(similarities)
    best_score = similarities[best_idx]

    # Confidence threshold — below this, use fallback
    if best_score < 0.15:
        return {
            "intent"    : "unknown",
            "confidence": best_score * 100,
            "response"  : get_fallback()
        }

    best_row = training_data.iloc[best_idx]
    confidence = min(99.9, best_score * 100 * 1.5)  # scale for display

    return {
        "intent"    : best_row["intent"],
        "confidence": confidence,
        "response"  : best_row["response"]
    }


def get_fallback() -> str:
    """Return a helpful fallback response."""
    fallbacks = [
        "I'm sorry, I didn't understand that. Could you rephrase your question?",
        "I'm not sure about that. Try asking about courses, internship, certificates, or submissions.",
        "That's outside my knowledge. Please contact contact@codevedx.in for further help.",
        "I couldn't find an answer. You can reach the team at teamleader@codevedx.in.",
    ]
    import random
    return random.choice(fallbacks)


# ============================================================
# SECTION 4 – Admin: Update FAQ
# ============================================================

def add_faq(intent: str, patterns: str, response: str):
    """Admin function to add new FAQ entry to dataset."""
    df = pd.read_csv(DATA_FILE)
    new_row = pd.DataFrame([{
        "intent"  : intent,
        "patterns": patterns,
        "response": response
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"  ✔ New FAQ added for intent: '{intent}'")
    print("  ⚠️  Please retrain the chatbot (Option 4) to apply changes.")


# ============================================================
# SECTION 5 – Chat History
# ============================================================

def save_history(history: list):
    """Save chat history to JSON file."""
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def load_history() -> list:
    """Load previous chat history."""
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)


# ============================================================
# SECTION 6 – Chat Interface
# ============================================================

def chat(model_data: dict):
    """Run the interactive chatbot session."""
    history = []

    print(f"\n{'='*58}")
    print("  🤖 CodeVedX AI HELPDESK CHATBOT")
    print("  Ask me anything about your internship!")
    print("  Type 'exit' to quit | Type 'history' to view chat log")
    print(f"{'='*58}\n")

    # Greeting
    print("  🤖 Bot: Hello! Welcome to CodeVedX Helpdesk.")
    print("         How can I help you today?\n")

    while True:
        user_input = input("  You: ").strip()

        if not user_input:
            continue

        # Special commands
        if user_input.lower() in ("exit", "quit", "bye"):
            print("\n  🤖 Bot: Thank you for using CodeVedX Helpdesk. Goodbye! 👋\n")
            save_history(history)
            break

        if user_input.lower() == "history":
            if history:
                print(f"\n  -- Chat History ({len(history)} messages) --")
                for h in history[-10:]:
                    print(f"  [{h['time']}] You: {h['user']}")
                    print(f"  [{h['time']}] Bot: {h['bot'][:80]}...")
                print()
            else:
                print("  No history yet.\n")
            continue

        # Detect intent and get response
        result = detect_intent(user_input, model_data)

        # Display response
        print(f"\n  🤖 Bot: {result['response']}")

        # Show intent info (for demo purposes)
        if result["intent"] != "unknown":
            print(f"  [Intent: {result['intent']} | Confidence: {result['confidence']:.1f}%]\n")
        else:
            print(f"  [Intent: unknown | Confidence: {result['confidence']:.1f}%]\n")

        # Save to history
        history.append({
            "time"      : datetime.now().strftime("%H:%M"),
            "user"      : user_input,
            "bot"       : result["response"],
            "intent"    : result["intent"],
            "confidence": round(result["confidence"], 1)
        })


# ============================================================
# SECTION 7 – Test All Intents
# ============================================================

def run_tests(model_data: dict):
    """Test all intents with sample queries."""
    test_cases = [
        ("hello there",                         "greeting"),
        ("what courses are available",           "course_info"),
        ("how do I apply for internship",        "internship_info"),
        ("when will I get my certificate",       "certificate"),
        ("how much does it cost",                "fees"),
        ("how long is the internship",           "duration"),
        ("how do I contact support",             "contact"),
        ("how to push code to github",           "github"),
        ("should I post on linkedin",            "linkedin"),
        ("how to submit my project",             "submission"),
        ("who is my mentor",                     "mentor"),
        ("join telegram group",                  "telegram"),
        ("what is the deadline",                 "deadline"),
        ("can I copy code from internet",        "plagiarism"),
        ("will I get a stipend",                 "stipend"),
        ("what skills will I learn",             "skills"),
        ("how many projects are there",          "project_count"),
        ("can I get a recommendation letter",    "recommendation"),
        ("goodbye thank you",                    "farewell"),
        ("what is the meaning of life",          "unknown"),
    ]

    print(f"\n{'='*65}")
    print("  CHATBOT INTENT DETECTION TEST")
    print(f"{'='*65}")
    print(f"  {'#':<3} {'Query':<38} {'Expected':<18} {'Got':<18} {'Conf%':<8} {'✔/✗'}")
    print("  " + "-"*85)

    correct = 0
    for i, (query, expected) in enumerate(test_cases, 1):
        result = detect_intent(query, model_data)
        got    = result["intent"]
        match  = "✔" if got == expected else "✗"
        if got == expected: correct += 1
        print(f"  {i:<3} {query[:36]:<38} {expected:<18} {got:<18} {result['confidence']:.1f}%   {match}")

    accuracy = correct / len(test_cases) * 100
    print(f"\n  Test Accuracy: {correct}/{len(test_cases)} = {accuracy:.0f}%\n")


# ============================================================
# SECTION 8 – Main Menu
# ============================================================

def main():
    print("\n  Welcome to the AI Helpdesk Chatbot!")
    model_data = load_chatbot()

    while True:
        print("""
╔══════════════════════════════════════════════╗
║   AI Chatbot for Internal Helpdesk           ║
║   CodeVedX AI/ML Internship – 2026           ║
╠══════════════════════════════════════════════╣
║  1. Start chat                               ║
║  2. Run intent detection tests               ║
║  3. Add new FAQ (Admin)                      ║
║  4. Retrain chatbot                          ║
║  5. View FAQ dataset                         ║
║  6. Exit                                     ║
╚══════════════════════════════════════════════╝""")

        choice = input("  Enter your choice (1-6): ").strip()

        if choice == "1":
            chat(model_data)

        elif choice == "2":
            run_tests(model_data)

        elif choice == "3":
            print("\n  -- Add New FAQ (Admin) --")
            intent   = input("  Intent name    : ").strip()
            patterns = input("  Patterns (comma-separated): ").strip()
            response = input("  Response       : ").strip()
            if intent and patterns and response:
                add_faq(intent, patterns, response)
            else:
                print("  All fields required.")

        elif choice == "4":
            model_data = train_chatbot()
            print("  ✔ Chatbot retrained successfully!")

        elif choice == "5":
            df = pd.read_csv(DATA_FILE)
            print(f"\n  FAQ Dataset — {len(df)} intents:")
            print(f"\n  {'#':<4} {'Intent':<22} {'Sample Pattern':<35} {'Response Preview'}")
            print("  " + "-"*90)
            for i, row in df.iterrows():
                sample = row["patterns"].split(",")[0].strip()
                print(f"  {i+1:<4} {row['intent']:<22} {sample:<35} {row['response'][:40]}...")
            print()

        elif choice == "6":
            print("\n  Goodbye! 👋\n")
            break

        else:
            print("  Invalid choice. Enter 1-6.")


if __name__ == "__main__":
    main()
