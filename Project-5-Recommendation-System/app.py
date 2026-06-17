from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

DATA_FILE = "courses.csv"


def load_courses():
    return pd.read_csv(DATA_FILE)


def prepare_course_data(courses):
    courses["combined_text"] = (
        courses["title"].fillna("") + " " +
        courses["category"].fillna("") + " " +
        courses["level"].fillna("") + " " +
        courses["skills"].fillna("") + " " +
        courses["description"].fillna("")
    )
    return courses


def get_recommendations(user_preference, top_n=10):
    courses = load_courses()
    courses = prepare_course_data(courses)

    documents = list(courses["combined_text"]) + [user_preference]

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(documents)

    user_vector = vectors[-1]
    course_vectors = vectors[:-1]

    similarity_scores = cosine_similarity(user_vector, course_vectors).flatten()

    courses["similarity_score"] = similarity_scores
    courses["match_percentage"] = (courses["similarity_score"] * 100).round(2)

    courses = courses[courses["similarity_score"] > 0]

    recommendations = courses.sort_values(
    by="similarity_score",
    ascending=False
    ).head(top_n)

    return recommendations.to_dict(orient="records")


@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    user_input = ""

    if request.method == "POST":
        user_input = request.form.get("preference", "").strip()

        if user_input:
            recommendations = get_recommendations(user_input)

    return render_template(
        "index.html",
        recommendations=recommendations,
        user_input=user_input
    )


if __name__ == "__main__":
    app.run(debug=True)