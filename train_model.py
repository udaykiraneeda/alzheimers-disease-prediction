
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------
# OPTION 1: can you please put the data set
# -----------------------------------------------------------------
CSV_PATH = "alzheimers_disease_data.csv"  # real dataset

FEATURE_COLUMNS = [
    'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking',
    'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
    'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes',
    'Depression', 'HeadInjury', 'Hypertension', 'SystolicBP', 'DiastolicBP',
    'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
    'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment',
    'MemoryComplaints', 'BehavioralProblems', 'ADL', 'Confusion',
    'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
    'Forgetfulness'
]
TARGET_COLUMN = "Diagnosis"


def load_data():
    if CSV_PATH:
        df = pd.read_csv(CSV_PATH)
        return df

    # -----------------------------------------------------------------
    # OPTION 2: Synthetic sample data (demo purpose only)
    # -----------------------------------------------------------------
    rng = np.random.default_rng(42)
    n = 1000

    data = {
        'Age': rng.integers(60, 91, n),
        'Gender': rng.integers(0, 2, n),
        'Ethnicity': rng.integers(0, 4, n),
        'EducationLevel': rng.integers(0, 4, n),
        'BMI': rng.uniform(15, 40, n),
        'Smoking': rng.integers(0, 2, n),
        'AlcoholConsumption': rng.uniform(0, 20, n),
        'PhysicalActivity': rng.uniform(0, 10, n),
        'DietQuality': rng.uniform(0, 10, n),
        'SleepQuality': rng.uniform(0, 10, n),
        'FamilyHistoryAlzheimers': rng.integers(0, 2, n),
        'CardiovascularDisease': rng.integers(0, 2, n),
        'Diabetes': rng.integers(0, 2, n),
        'Depression': rng.integers(0, 2, n),
        'HeadInjury': rng.integers(0, 2, n),
        'Hypertension': rng.integers(0, 2, n),
        'SystolicBP': rng.integers(90, 181, n),
        'DiastolicBP': rng.integers(60, 121, n),
        'CholesterolTotal': rng.uniform(100, 300, n),
        'CholesterolLDL': rng.uniform(50, 200, n),
        'CholesterolHDL': rng.uniform(20, 100, n),
        'CholesterolTriglycerides': rng.uniform(50, 400, n),
        'MMSE': rng.uniform(0, 30, n),
        'FunctionalAssessment': rng.uniform(0, 10, n),
        'MemoryComplaints': rng.integers(0, 2, n),
        'BehavioralProblems': rng.integers(0, 2, n),
        'ADL': rng.uniform(0, 10, n),
        'Confusion': rng.integers(0, 2, n),
        'Disorientation': rng.integers(0, 2, n),
        'PersonalityChanges': rng.integers(0, 2, n),
        'DifficultyCompletingTasks': rng.integers(0, 2, n),
        'Forgetfulness': rng.integers(0, 2, n),
    }
    df = pd.DataFrame(data)

    # Synthetic "risk score" logic based on known Alzheimer's indicators
    # (low MMSE, low FunctionalAssessment, high age, memory complaints etc.
    # increase risk). Its just demo label generation.
    risk = (
        (30 - df['MMSE']) * 0.15
        + (10 - df['FunctionalAssessment']) * 0.2
        + (df['Age'] - 60) * 0.05
        + df['MemoryComplaints'] * 1.5
        + df['BehavioralProblems'] * 1.0
        + df['Confusion'] * 1.0
        + df['Disorientation'] * 1.0
        + df['FamilyHistoryAlzheimers'] * 0.8
        + (10 - df['ADL']) * 0.15
        + rng.normal(0, 1.5, n)
    )
    threshold = np.percentile(risk, 60)
    df[TARGET_COLUMN] = (risk > threshold).astype(int)
    return df


def main():
    df = load_data()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")

    joblib.dump(model, "best_logistic_regression_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\nSaved: best_logistic_regression_model.pkl, scaler.pkl")


if __name__ == "__main__":
    main()
