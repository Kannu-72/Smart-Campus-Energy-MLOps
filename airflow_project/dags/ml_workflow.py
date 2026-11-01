from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "feature_store" / "retrieved_features.csv"
DEPLOYMENT_DIR = BASE_DIR / "deployment"

MODEL_PATH = DEPLOYMENT_DIR / "exp6_orchestrated_model.pkl"
RESULT_PATH = DEPLOYMENT_DIR / "exp6_results.json"


# STEP 1: Load and validate dataset
def load_validate_data():

    print("STEP 1: Loading Smart Campus Energy Dataset")

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "occupancy", "temperature", "humidity", "co2_level",
        "hvac_status", "lighting_status", "outdoor_temperature",
        "building_id", "room_type",
        "hour", "day_of_week", "is_weekend",
        "occupancy_per_co2", "thermal_gap",
        "hvac_occupancy_interaction",
        "avg_occupancy", "max_co2", "hvac_usage_rate",
        "high_energy_usage"
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)

    return str(DATA_PATH)


# STEP 2: Train and evaluate model
def train_evaluate_model(data_path):

    print("\nSTEP 2: Preparing features")

    df = pd.read_csv(data_path)

    numeric_features = [
        "occupancy", "temperature", "humidity", "co2_level",
        "hvac_status", "lighting_status", "outdoor_temperature",
        "hour", "day_of_week", "is_weekend",
        "occupancy_per_co2", "thermal_gap",
        "hvac_occupancy_interaction",
        "avg_occupancy", "max_co2", "hvac_usage_rate"
    ]

    categorical_features = ["building_id", "room_type"]

    X = df[numeric_features + categorical_features].copy()
    y = df["high_energy_usage"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # Numeric preprocessing
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # Complete ML pipeline
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(
            C=10,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

    print("\nSTEP 3: Training Logistic Regression")

    model.fit(X_train, y_train)

    print("Model training completed!")

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(
            y_test, y_pred, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, y_pred, zero_division=0
        )),
        "f1_score": float(f1_score(
            y_test, y_pred, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }

    print("\nSTEP 4: Model Evaluation")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # Save trained model
    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved at:", MODEL_PATH)

    return metrics


# STEP 5: Save evaluation results
def save_results(metrics):

    with open(RESULT_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nEvaluation results saved at:", RESULT_PATH)

    return "ML workflow completed successfully!"