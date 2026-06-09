# Project 2 – Student Performance Prediction System

**CodeVedX AI/ML Internship 2026 | Akanksha Sudabattula**

---

## Overview

A machine learning system that predicts a student's final grade based on their attendance, subject marks, study habits, and lifestyle factors. Includes full **Exploratory Data Analysis (EDA)** and **data visualizations**.

---

## Features

| # | Feature |
|---|---------|
| 1 | View dataset overview (shape, missing values, sample rows) |
| 2 | Exploratory Data Analysis — stats, correlations, grade distribution |
| 3 | Data visualization — 4 charts (scatter, bar, trend, heatmap) |
| 4 | Add new student records to the dataset |
| 5 | Train ML model & predict final grade with personalized feedback |

---

## Tech Stack

- **Language** – Python 3.x
- **Libraries** – Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Models** – Linear Regression + Random Forest Regressor
- **Storage** – CSV file (`student_data.csv`)

---

## How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Run the system
```bash
python student_performance.py
```

---

## Input Features

| Feature | Description |
|---------|-------------|
| `attendance_percent` | Student's attendance percentage |
| `math_marks` | Math subject marks (0-100) |
| `science_marks` | Science subject marks (0-100) |
| `english_marks` | English subject marks (0-100) |
| `study_hours_per_day` | Daily study hours |
| `sleep_hours` | Daily sleep hours |
| `extracurricular` | Participates in activities (0/1) |

## Output

- Predicted final grade (0–100)
- Grade band (A/B/C/D/F)
- Personalised improvement suggestions

---

## Model Results

| Model | MAE | R² Score |
|-------|-----|----------|
| Linear Regression | 0.16 | 0.9998 |
| Random Forest | 0.67 | 0.9968 |

---

## Visualizations Generated

- Study Hours vs Final Grade (scatter + attendance color)
- Average Marks by Subject (bar chart)
- Attendance % vs Final Grade (scatter + trend line)
- Feature Correlation Heatmap

---

## File Structure

```
Project-2-Student-Performance-Prediction/
├── student_performance.py   ← main application
├── student_data.csv         ← dataset (40 student records)
└── README.md                ← this file
```

---

*Built as part of CodeVedX AI/ML Virtual Internship – Batch 2026*