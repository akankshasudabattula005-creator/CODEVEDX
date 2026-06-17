import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "courses.csv"


def load_courses():
    return pd.read_csv(DATA_FILE)


def build_course_text(data):
    data["combined_features"] = (
        data["title"].fillna("") + " " +
        data["category"].fillna("") + " " +
        data["level"].fillna("") + " " +
        data["skills"].fillna("") + " " +
        data["description"].fillna("")
    )
    return data


def recommend_courses(user_preferences, top_n=5):
    courses = load_courses()
    courses = build_course_text(courses)

    all_text = list(courses["combined_features"]) + [user_preferences]

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(all_text)

    user_vector = vectors[-1]
    course_vectors = vectors[:-1]

    similarity_scores = cosine_similarity(user_vector, course_vectors).flatten()

    courses["similarity_score"] = similarity_scores
    recommendations = courses.sort_values(
        by="similarity_score",
        ascending=False
    ).head(top_n)

    return recommendations


def main():
    print("\n===== Smart Course Recommendation System =====")
    print("Enter your interests, skills, or learning goals.")
    print("Example: python machine learning beginner data analysis\n")

    user_input = input("Your Preference: ")

    if not user_input.strip():
        print("Please enter at least one preference.")
        return

    recommendations = recommend_courses(user_input)

    print("\nRecommended Courses For You:\n")

    for index, row in recommendations.iterrows():
        print(f"{row['title']}")
        print(f"Category: {row['category']}")
        print(f"Level: {row['level']}")
        print(f"Match Score: {row['similarity_score']:.2f}")
        print("-" * 40)


if __name__ == "__main__":
    main()