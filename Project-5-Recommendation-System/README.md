# Smart Course Recommendation System

An AI-powered Course Recommendation System developed as part of the **CodeVedX AI/ML Internship**. The application recommends the most relevant courses based on user interests using **Content-Based Filtering**, **TF-IDF Vectorization**, and **Cosine Similarity**.

---

## 📌 Project Overview

This project helps users discover courses that match their learning interests and career goals. By analyzing the text entered by the user and comparing it with course information, the system provides personalized recommendations ranked by similarity score.

The project includes both:

- 💻 Console-based recommendation system
- 🌐 Flask-based web application with a modern user interface

---

## ✨ Features

- User preference input
- Personalized course recommendations
- Content-based filtering
- TF-IDF feature extraction
- Cosine similarity scoring
- Match percentage display
- Responsive web interface
- Clean and modular project structure

---

## 🛠️ Technologies Used

- Python
- Flask
- Pandas
- Scikit-learn
- HTML
- CSS

---

## 🧠 Machine Learning Technique

The recommendation engine uses **Content-Based Filtering**.

### Workflow

1. Load course dataset
2. Combine course title, category, level, skills, and description
3. Convert text into numerical vectors using TF-IDF
4. Calculate similarity using Cosine Similarity
5. Rank courses based on similarity score
6. Display the best matching recommendations

---

## 📂 Project Structure

```
Project-5-Recommendation-System/
│
├── app.py
├── recommendation_system.py
├── courses.csv
├── requirements.txt
├── README.md
│
└── templates/
    └── index.html
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone <repository-url>
```

Move to the project directory:

```bash
cd Project-5-Recommendation-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Web Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 📝 Sample User Input

```
python machine learning beginner data analysis
```

---

## 📊 Sample Recommendation Output

- Machine Learning Basics
- Data Analysis with Pandas
- Python for Beginners
- Deep Learning Introduction
- SQL for Data Analytics

Each recommendation includes a similarity-based match percentage.

---

## 🔮 Future Enhancements

- User authentication
- Recommendation history
- Favorite courses
- Category-based filtering
- Explainable AI recommendations
- Database integration
- Cloud deployment
- Hybrid recommendation engine

---

## 🎯 Learning Outcomes

This project demonstrates practical implementation of:

- Machine Learning
- Natural Language Processing
- Recommendation Systems
- Flask Web Development
- Data Processing with Pandas
- Similarity Algorithms
- AI-based Personalization

---

## 👩‍💻 Author

**Akanksha Sudabattula**

Developed as part of the **CodeVedX AI/ML Internship 2026**.

---