# ============================================================
#  Student Performance Prediction System
#  Author : Akanksha Sudabattula
#  Internship : CodeVedX – AI/ML Batch 2026
#  Project 2  : ML + Data Analysis + Visualization
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings("ignore")

# ---------- File path ----------
DATA_FILE = "student_data.csv"

# ============================================================
# SECTION 1 – Data Loading & Cleaning
# ============================================================

def load_data():
    """Load CSV, handle missing values, return clean DataFrame."""
    if not os.path.exists(DATA_FILE):
        print(f"  [ERROR] '{DATA_FILE}' not found. Please place it in the same folder.")
        return None

    df = pd.read_csv(DATA_FILE)

    # ---- Handle missing values ----
    # Fill numeric columns with their median (robust to outliers)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"  [INFO] Filled {missing} missing values in '{col}' with median.")

    return df


def show_data_info(df):
    """Display basic info about the dataset."""
    print(f"\n{'='*60}")
    print("  DATASET OVERVIEW")
    print(f"{'='*60}")
    print(f"  Total students  : {len(df)}")
    print(f"  Total features  : {len(df.columns)}")
    print(f"  Missing values  : {df.isnull().sum().sum()}")
    print(f"\n  Columns: {list(df.columns)}")
    print(f"\n  First 5 rows:")
    print(df.head().to_string(index=False))
    print()


# ============================================================
# SECTION 2 – Input Helpers
# ============================================================

def get_float(prompt, min_val=0.0, max_val=9999.0):
    """Get a validated float input from user."""
    while True:
        try:
            val = float(input(prompt))
            if min_val <= val <= max_val:
                return val
            print(f"  Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("  Invalid input – please enter a number.")


def get_int(prompt, min_val=0, max_val=9999):
    """Get a validated integer input from user."""
    while True:
        try:
            val = int(input(prompt))
            if min_val <= val <= max_val:
                return val
            print(f"  Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("  Invalid input – please enter a whole number.")


# ============================================================
# SECTION 3 – Exploratory Data Analysis (EDA)
# ============================================================

def exploratory_analysis(df):
    """Menu Option 2 – Show descriptive stats and data insights."""
    print(f"\n{'='*60}")
    print("  EXPLORATORY DATA ANALYSIS")
    print(f"{'='*60}")

    print("\n  📊 Descriptive Statistics:")
    print(df.describe().round(2).to_string())

    print(f"\n  📈 Average scores by subject:")
    for col in ["math_marks", "science_marks", "english_marks"]:
        print(f"     {col:<20} : {df[col].mean():.2f}")

    print(f"\n  🎯 Final grade distribution:")
    bins   = [0, 40, 55, 70, 85, 100]
    labels = ["Fail (<40)", "D (40-55)", "C (55-70)", "B (70-85)", "A (85+)"]
    df["grade_band"] = pd.cut(df["final_grade"], bins=bins, labels=labels)
    print(df["grade_band"].value_counts().to_string())
    df.drop(columns=["grade_band"], inplace=True)

    print(f"\n  📌 Correlation with final_grade:")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()["final_grade"].drop("final_grade").sort_values(ascending=False)
    for feat, val in corr.items():
        bar = "█" * int(abs(val) * 20)
        print(f"     {feat:<25} : {val:+.3f}  {bar}")
    print()


# ============================================================
# SECTION 4 – Data Visualization
# ============================================================

def visualize_data(df):
    """Menu Option 3 – Generate 4 charts and save as PNG."""
    print("\n  Generating visualizations... please wait.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Student Performance Analysis", fontsize=16, fontweight="bold", y=1.01)
    plt.subplots_adjust(hspace=0.4, wspace=0.35)

    # ---- Chart 1: Study Hours vs Final Grade (scatter) ----
    ax1 = axes[0, 0]
    scatter = ax1.scatter(
        df["study_hours_per_day"], df["final_grade"],
        c=df["attendance_percent"], cmap="viridis", alpha=0.75, s=60
    )
    plt.colorbar(scatter, ax=ax1, label="Attendance %")
    ax1.set_xlabel("Study Hours per Day")
    ax1.set_ylabel("Final Grade")
    ax1.set_title("Study Hours vs Final Grade\n(color = Attendance %)")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # ---- Chart 2: Average subject marks bar chart ----
    ax2 = axes[0, 1]
    subjects = ["math_marks", "science_marks", "english_marks"]
    averages = [df[s].mean() for s in subjects]
    colors   = ["#028090", "#02C39A", "#0D3349"]
    bars = ax2.bar(["Math", "Science", "English"], averages, color=colors, width=0.5)
    for bar, avg in zip(bars, averages):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{avg:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Average Marks")
    ax2.set_title("Average Marks by Subject")
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # ---- Chart 3: Attendance vs Final Grade (scatter with trend) ----
    ax3 = axes[1, 0]
    ax3.scatter(df["attendance_percent"], df["final_grade"],
                color="#028090", alpha=0.65, s=55)
    # Add trend line
    z = np.polyfit(df["attendance_percent"], df["final_grade"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["attendance_percent"].min(), df["attendance_percent"].max(), 100)
    ax3.plot(x_line, p(x_line), color="#F96167", linewidth=2, linestyle="--", label="Trend")
    ax3.set_xlabel("Attendance %")
    ax3.set_ylabel("Final Grade")
    ax3.set_title("Attendance % vs Final Grade")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.4)

    # ---- Chart 4: Correlation heatmap ----
    ax4 = axes[1, 1]
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    sns.heatmap(
        corr_matrix, ax=ax4, annot=True, fmt=".2f",
        cmap="coolwarm", linewidths=0.5,
        annot_kws={"size": 7}, square=True
    )
    ax4.set_title("Feature Correlation Heatmap")
    ax4.tick_params(axis="x", rotation=45, labelsize=7)
    ax4.tick_params(axis="y", rotation=0,  labelsize=7)

    plt.tight_layout()
    chart_path = "student_analysis_charts.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  ✔ Charts saved as '{chart_path}'\n")


# ============================================================
# SECTION 5 – Feature Engineering & ML Model
# ============================================================

def prepare_features(df):
    """Create feature matrix X and target vector y."""
    feature_cols = [
        "attendance_percent",
        "math_marks",
        "science_marks",
        "english_marks",
        "study_hours_per_day",
        "sleep_hours",
        "extracurricular"
    ]
    X = df[feature_cols].values
    y = df["final_grade"].values
    return X, y, feature_cols


def train_models(df):
    """
    Train two models — Linear Regression and Random Forest.
    Compare their performance and return the better one.
    """
    X, y, feature_cols = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression" : LinearRegression(),
        "Random Forest"     : RandomForestRegressor(n_estimators=100, random_state=42)
    }

    print(f"\n{'='*60}")
    print("  MODEL TRAINING & EVALUATION")
    print(f"{'='*60}")
    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing  samples : {len(X_test)}")
    print(f"{'='*60}")
    print(f"  {'Model':<22} | {'MAE':>6} | {'RMSE':>6} | {'R² Score':>9}")
    print(f"  {'-'*52}")

    best_model      = None
    best_r2         = -999
    best_model_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        print(f"  {name:<22} | {mae:>6.2f} | {rmse:>6.2f} | {r2:>9.4f}")

        if r2 > best_r2:
            best_r2         = r2
            best_model      = model
            best_model_name = name

    print(f"{'='*60}")
    print(f"  ✔ Best model: {best_model_name} (R² = {best_r2:.4f})\n")

    # Feature importance (only for Random Forest)
    if best_model_name == "Random Forest":
        importances = best_model.feature_importances_
        print("  📌 Feature Importances:")
        for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
            bar = "█" * int(imp * 40)
            print(f"     {feat:<25} : {imp:.4f}  {bar}")
        print()

    return best_model, feature_cols


def predict_student(model, feature_cols):
    """Menu Option 5 – Predict a student's final grade from user input."""
    print("\n--- Enter Student Details for Prediction ---")

    attendance   = get_float("  Attendance percentage (0-100)  : ", 0, 100)
    math         = get_float("  Math marks (0-100)             : ", 0, 100)
    science      = get_float("  Science marks (0-100)          : ", 0, 100)
    english      = get_float("  English marks (0-100)          : ", 0, 100)
    study_hours  = get_float("  Study hours per day (0-24)     : ", 0, 24)
    sleep_hours  = get_float("  Sleep hours per day (0-24)     : ", 0, 24)
    extra        = get_int  ("  Extracurricular activities (0=No, 1=Yes): ", 0, 1)

    user_input = np.array([[attendance, math, science, english,
                            study_hours, sleep_hours, extra]])

    predicted_grade = model.predict(user_input)[0]
    predicted_grade = max(0, min(100, predicted_grade))  # clamp to 0-100

    # Determine grade band
    if predicted_grade >= 85:
        band, emoji = "A (Excellent)", "🌟"
    elif predicted_grade >= 70:
        band, emoji = "B (Good)",      "✅"
    elif predicted_grade >= 55:
        band, emoji = "C (Average)",   "📘"
    elif predicted_grade >= 40:
        band, emoji = "D (Below Average)", "⚠️"
    else:
        band, emoji = "F (Fail)",      "❌"

    print(f"\n{'='*60}")
    print("  PREDICTION RESULT")
    print(f"{'='*60}")
    print(f"  Predicted Final Grade : {predicted_grade:.2f} / 100")
    print(f"  Grade Band            : {band} {emoji}")
    print(f"{'='*60}")

    # Personalised feedback
    print("\n  💡 Suggestions:")
    if attendance < 75:
        print("     • Attendance is low — aim for at least 75%")
    if study_hours < 3:
        print("     • Increase study hours to at least 3 hours/day")
    if sleep_hours < 6:
        print("     • Getting more sleep improves performance")
    if math < 50 or science < 50 or english < 50:
        print("     • Focus on weaker subjects with extra practice")
    if predicted_grade >= 85:
        print("     • Excellent performance! Keep it up 🎉")
    print()


# ============================================================
# SECTION 6 – Add New Student Record
# ============================================================

def add_student(df):
    """Menu Option 4 – Add a new student record to the CSV."""
    print("\n--- Add New Student Record ---")

    new_id = df["student_id"].max() + 1
    attendance  = get_float("  Attendance percentage (0-100)  : ", 0, 100)
    math        = get_float("  Math marks (0-100)             : ", 0, 100)
    science     = get_float("  Science marks (0-100)          : ", 0, 100)
    english     = get_float("  English marks (0-100)          : ", 0, 100)
    study_hours = get_float("  Study hours per day            : ", 0, 24)
    sleep_hours = get_float("  Sleep hours per day            : ", 0, 24)
    extra       = get_int  ("  Extracurricular (0=No, 1=Yes)  : ", 0, 1)
    final_grade = get_float("  Final grade (0-100)            : ", 0, 100)

    new_row = {
        "student_id"          : int(new_id),
        "attendance_percent"  : attendance,
        "math_marks"          : math,
        "science_marks"       : science,
        "english_marks"       : english,
        "study_hours_per_day" : study_hours,
        "sleep_hours"         : sleep_hours,
        "extracurricular"     : extra,
        "final_grade"         : final_grade
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"  ✔ Student {int(new_id)} added successfully! Total records: {len(df)}\n")
    return df


# ============================================================
# SECTION 7 – Main Menu
# ============================================================

def print_menu():
    print("""
╔══════════════════════════════════════════════╗
║   Student Performance Prediction System      ║
║   CodeVedX AI/ML Internship – 2026           ║
╠══════════════════════════════════════════════╣
║  1. View dataset overview                    ║
║  2. Exploratory Data Analysis (EDA)          ║
║  3. Visualize data (charts)                  ║
║  4. Add new student record                   ║
║  5. Train model & predict student grade      ║
║  6. Exit                                     ║
╚══════════════════════════════════════════════╝""")


def main():
    print("\n  Welcome to the Student Performance Prediction System!")

    df = load_data()
    if df is None:
        return

    trained_model  = None
    trained_feats  = None

    while True:
        print_menu()
        choice = input("  Enter your choice (1-6): ").strip()

        if choice == "1":
            show_data_info(df)

        elif choice == "2":
            exploratory_analysis(df)

        elif choice == "3":
            visualize_data(df)

        elif choice == "4":
            df = add_student(df)

        elif choice == "5":
            if trained_model is None:
                trained_model, trained_feats = train_models(df)
            predict_student(trained_model, trained_feats)

        elif choice == "6":
            print("\n  Thank you! Goodbye 👋\n")
            break

        else:
            print("\n  Invalid choice. Please enter a number between 1 and 6.\n")


# ---- Entry point ----
if __name__ == "__main__":
    main()