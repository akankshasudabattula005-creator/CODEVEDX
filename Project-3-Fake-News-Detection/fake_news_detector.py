# ============================================================
#  AI Based Fake News Detection Tool
#  Author : Akanksha Sudabattula
#  Internship : CodeVedX – AI/ML Batch 2026
#  Project 3  : NLP + Text Classification + ML Model Storage
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle, os, re
import warnings
warnings.filterwarnings("ignore")

DATA_FILE  = "news_data.csv"
MODEL_FILE = "fake_news_model.pkl"
TFIDF_FILE = "tfidf_vectorizer.pkl"


# ============================================================
# SECTION 1 – Text Preprocessing (Tokenization + Cleaning)
# ============================================================

def preprocess_text(text: str) -> str:
    """
    NLP Preprocessing steps:
    1. Lowercase conversion
    2. Remove special characters and numbers
    3. Remove extra whitespace
    4. Remove short words (stopword filtering)
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if len(w) >= 3]
    return " ".join(words)


# ============================================================
# SECTION 2 – Model Training
# ============================================================

def train_model():
    """
    Train Fake News Detection model:
    - TF-IDF Vectorization (text → numerical features)
    - PassiveAggressiveClassifier (best for text classification)
    - Save model and vectorizer to disk
    """
    df = pd.read_csv(DATA_FILE)
    print(f"\n  Dataset: {len(df)} samples | REAL: {(df['label']=='REAL').sum()} | FAKE: {(df['label']=='FAKE').sum()}")

    df["clean_text"] = df["text"].apply(preprocess_text)

    # TF-IDF: converts text to numerical matrix
    # ngram_range=(1,2) means single words AND word pairs
    tfidf = TfidfVectorizer(
        max_features = 5000,
        ngram_range  = (1, 2),
        stop_words   = "english",
        sublinear_tf = True
    )

    X = tfidf.fit_transform(df["clean_text"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = PassiveAggressiveClassifier(max_iter=500, C=1.0, random_state=42)
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*55}")
    print("  MODEL TRAINING RESULTS")
    print(f"{'='*55}")
    print(f"  Training samples : {X_train.shape[0]}")
    print(f"  Testing  samples : {X_test.shape[0]}")
    print(f"  Accuracy         : {accuracy*100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["REAL","FAKE"]))

    with open(MODEL_FILE, "wb") as f: pickle.dump(model, f)
    with open(TFIDF_FILE, "wb") as f: pickle.dump(tfidf, f)
    print(f"  ✔ Model saved as '{MODEL_FILE}'")
    print(f"  ✔ Vectorizer saved as '{TFIDF_FILE}'\n")
    return model, tfidf, accuracy


def load_model():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(TFIDF_FILE):
        print("  [INFO] No saved model found. Training now...")
        return train_model()[:2]
    with open(MODEL_FILE, "rb") as f: model = pickle.load(f)
    with open(TFIDF_FILE, "rb") as f: tfidf = pickle.load(f)
    print("  ✔ Model loaded from disk.")
    return model, tfidf


# ============================================================
# SECTION 3 – Prediction with Confidence Score
# ============================================================

def predict_news(text: str, model, tfidf) -> dict:
    """
    Predict FAKE or REAL with confidence score.
    Uses decision_function distance from boundary as confidence.
    """
    clean  = preprocess_text(text)
    vector = tfidf.transform([clean])
    label  = model.predict(vector)[0]

    # decision_function gives distance from boundary
    # larger distance = higher confidence
    score      = abs(model.decision_function(vector)[0])
    confidence = min(99.9, 50 + score * 25)  # scale to 50-99%

    if confidence >= 90: level = "Very High"
    elif confidence >= 75: level = "High"
    elif confidence >= 60: level = "Moderate"
    else: level = "Low"

    fake_conf = confidence if label == "FAKE" else 100 - confidence
    real_conf = 100 - fake_conf

    return {
        "label"      : label,
        "confidence" : confidence,
        "level"      : level,
        "real_prob"  : real_conf,
        "fake_prob"  : fake_conf,
        "warning"    : "⚠️  Low confidence — verify independently." if confidence < 60 else ""
    }


def display_prediction(result, text):
    emoji    = "🚨 FAKE" if result["label"] == "FAKE" else "✅ REAL"
    bar_len  = int(result["confidence"] / 5)
    color_bar = "█" * bar_len + "░" * (20 - bar_len)

    print(f"\n{'='*58}")
    print("  DETECTION RESULT")
    print(f"{'='*58}")
    print(f"  News       : {text[:55]}{'...' if len(text)>55 else ''}")
    print(f"  Result     : {emoji} NEWS")
    print(f"  Confidence : {result['confidence']:.1f}% ({result['level']})")
    print(f"  [{color_bar}]")
    print(f"\n  REAL probability : {result['real_prob']:.1f}%")
    print(f"  FAKE probability : {result['fake_prob']:.1f}%")
    if result["warning"]:
        print(f"\n  {result['warning']}")
    print(f"{'='*58}\n")


# ============================================================
# SECTION 4 – Visualization
# ============================================================

def visualize_results(model, tfidf):
    df = pd.read_csv(DATA_FILE)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Fake News Detection — Analysis", fontsize=15, fontweight="bold")

    # Chart 1: Dataset distribution
    counts = df["label"].value_counts()
    axes[0].bar(counts.index, counts.values, color=["#02C39A","#F96167"], width=0.5)
    axes[0].set_title("Dataset Distribution")
    axes[0].set_ylabel("Samples")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v+0.3, str(v), ha="center", fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Chart 2: Top fake news keywords
    from collections import Counter
    fake_words = " ".join(df[df["label"]=="FAKE"]["text"].apply(preprocess_text)).split()
    stopextra  = {"the","and","that","this","from","with","have","will","are","for","all","new","not","but","has"}
    top = dict(Counter(w for w in fake_words if w not in stopextra).most_common(8))
    axes[1].barh(list(top.keys()), list(top.values()), color="#F96167", height=0.6)
    axes[1].set_title("Top Words in Fake News")
    axes[1].set_xlabel("Frequency")
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    # Chart 3: Sample confidence scores
    samples = [
        "Scientists discover new vaccine effective against virus",
        "Aliens landed government hiding the truth from everyone",
        "Government launches new education policy for rural areas",
        "Bill Gates microchipping people through vaccines secret",
        "New renewable energy plant powers 50000 homes next year",
        "Secret society controls all world governments confirmed",
    ]
    s_labels = [s[:22]+"..." for s in samples]
    s_conf   = [predict_news(s, model, tfidf)["confidence"] for s in samples]
    s_colors = [predict_news(s, model, tfidf)["label"] for s in samples]
    colors   = ["#F96167" if c=="FAKE" else "#02C39A" for c in s_colors]

    axes[2].barh(s_labels, s_conf, color=colors, height=0.6)
    axes[2].set_xlabel("Confidence (%)")
    axes[2].set_title("Sample Predictions")
    axes[2].set_xlim(0, 110)
    axes[2].grid(axis="x", linestyle="--", alpha=0.4)
    for i, v in enumerate(s_conf):
        axes[2].text(v+1, i, f"{v:.0f}%", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("fake_news_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔ Charts saved as 'fake_news_analysis.png'\n")


# ============================================================
# SECTION 5 – Batch Test
# ============================================================

def batch_test(model, tfidf):
    samples = [
        ("Scientists discover new vaccine effective against viruses",    "REAL"),
        ("Government hiding alien contact made 50 years ago proof",      "FAKE"),
        ("Annual budget allocates funds for public healthcare system",   "REAL"),
        ("Miracle cure for cancer suppressed by pharmaceutical firms",   "FAKE"),
        ("New renewable energy project to power 100000 homes next year", "REAL"),
        ("Secret society controls world governments banks and media",     "FAKE"),
        ("University research shows sleep affects memory performance",    "REAL"),
        ("Celebrities arrested for running underground criminal network", "FAKE"),
    ]
    print(f"\n{'='*65}")
    print("  BATCH TEST RESULTS")
    print(f"{'='*65}")
    print(f"  {'#':<3} {'Predicted':<8} {'Actual':<8} {'Conf%':<8} {'Match':<6} {'Headline'}")
    print("  " + "-"*65)

    correct = 0
    for i, (text, true_label) in enumerate(samples, 1):
        r = predict_news(text, model, tfidf)
        match = "✔" if r["label"] == true_label else "✗"
        if r["label"] == true_label: correct += 1
        print(f"  {i:<3} {r['label']:<8} {true_label:<8} {r['confidence']:.1f}%   {match}    {text[:38]}...")

    print(f"\n  Batch Accuracy: {correct}/{len(samples)} = {correct/len(samples)*100:.0f}%\n")


# ============================================================
# SECTION 6 – Main Menu
# ============================================================

def main():
    print("\n  Welcome to the AI Fake News Detection Tool!")
    print("  Loading model...")
    model, tfidf = load_model()

    while True:
        print("""
╔══════════════════════════════════════════════╗
║   AI Based Fake News Detection Tool         ║
║   CodeVedX AI/ML Internship – 2026          ║
╠══════════════════════════════════════════════╣
║  1. Detect single news article              ║
║  2. Train / retrain model                   ║
║  3. Batch test (8 samples)                  ║
║  4. Generate analysis charts                ║
║  5. Exit                                    ║
╚══════════════════════════════════════════════╝""")
        choice = input("  Enter your choice (1-5): ").strip()

        if choice == "1":
            text = input("\n  Enter news headline: ").strip()
            if text:
                display_prediction(predict_news(text, model, tfidf), text)
        elif choice == "2":
            model, tfidf, acc = train_model()
            print(f"  ✔ Accuracy: {acc*100:.2f}%")
        elif choice == "3":
            batch_test(model, tfidf)
        elif choice == "4":
            visualize_results(model, tfidf)
        elif choice == "5":
            print("\n  Goodbye! 👋\n")
            break
        else:
            print("  Invalid choice.")

if __name__ == "__main__":
    main()
