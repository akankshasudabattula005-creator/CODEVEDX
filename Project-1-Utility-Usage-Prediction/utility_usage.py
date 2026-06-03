import pandas as pd
from sklearn.linear_model import LinearRegression

FILE_NAME = "utility_data.csv"

def add_data():
    try:
        month = int(input("Enter Month: "))
        usage = float(input("Enter Usage: "))

        new_data = pd.DataFrame({
            "month": [month],
            "usage": [usage]
        })

        new_data.to_csv(FILE_NAME, mode='a', header=False, index=False)
        print("Data Added Successfully!")

    except Exception as e:
        print("Error:", e)

def view_data():
    try:
        data = pd.read_csv(FILE_NAME)
        print("\nUtility Usage Data:")
        print(data)

    except Exception as e:
        print("Error:", e)

def predict_usage():
    try:
        data = pd.read_csv(FILE_NAME)

        X = data[["month"]]
        y = data["usage"]

        model = LinearRegression()
        model.fit(X, y)

        future_month = int(input("Enter Future Month: "))
        future_data = pd.DataFrame({
            "month": [future_month]
       })
        prediction = model.predict(future_data)

        print(f"Predicted Usage: {prediction[0]:.2f}")

    except Exception as e:
        print("Error:", e)

while True:
    print("\n===== Utility Usage Prediction Tool =====")
    print("1. Add Data")
    print("2. View Data")
    print("3. Predict Usage")
    print("4. Exit")

    choice = input("Enter Choice: ")

    if choice == "1":
        add_data()

    elif choice == "2":
        view_data()

    elif choice == "3":
        predict_usage()

    elif choice == "4":
        print("Exiting Program...")
        break

    else:
        print("Invalid Choice!")