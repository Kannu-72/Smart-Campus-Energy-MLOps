
from flask import Flask, jsonify, request
from pathlib import Path
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)

# Deployment artifact folder
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "deployment"

# Load saved artifacts
preprocessor = joblib.load(
    MODEL_DIR / "preprocessing_pipeline.pkl"
)

best_model = joblib.load(
    MODEL_DIR / "best_smart_campus_model.pkl"
)

eng_imputer = joblib.load(
    MODEL_DIR / "engineered_features_imputer.pkl"
)

print("All model artifacts loaded successfully!")


# Homepage
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "service": "smart-campus-energy-prediction"
    })


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "JSON input required"}), 400

        original_features = [
            "occupancy",
            "temperature",
            "humidity",
            "co2_level",
            "hvac_status",
            "lighting_status",
            "outdoor_temperature",
            "building_id",
            "room_type"
        ]

        engineered_features = [
            "hour",
            "day_of_week",
            "is_weekend",
            "occupancy_per_co2",
            "thermal_gap",
            "hvac_occupancy_interaction",
            "avg_occupancy",
            "max_co2",
            "hvac_usage_rate"
        ]

        required_features = original_features + engineered_features

        missing = [
            feature for feature in required_features
            if feature not in data
        ]

        if missing:
            return jsonify({
                "error": "Missing input features",
                "missing_features": missing
            }), 400

        # Prepare original features
        original_df = pd.DataFrame(
            [[data[f] for f in original_features]],
            columns=original_features
        )

        # Prepare engineered features
        engineered_df = pd.DataFrame(
            [[data[f] for f in engineered_features]],
            columns=engineered_features
        )

        # Apply saved preprocessing
        original_processed = preprocessor.transform(original_df)

        engineered_processed = eng_imputer.transform(
            engineered_df
        )

        # Combine both feature sets
        final_features = hstack([
            csr_matrix(original_processed),
            csr_matrix(engineered_processed)
        ]).tocsr()

        # Predict
        prediction = int(best_model.predict(final_features)[0])

        response = {
            "prediction": prediction,
            "result": (
                "High Energy Usage"
                if prediction == 1
                else "Normal Energy Usage"
            )
        }

        if hasattr(best_model, "predict_proba"):
            probabilities = best_model.predict_proba(final_features)[0]
            classes = list(best_model.classes_)

            response["probability_normal"] = float(
                probabilities[classes.index(0)]
            )

            response["probability_high_usage"] = float(
                probabilities[classes.index(1)]
            )

        return jsonify(response), 200

    except (TypeError, ValueError) as error:
        return jsonify({
            "error": "Invalid feature values",
            "details": str(error)
        }), 400

    except Exception as error:
        return jsonify({
            "error": "Prediction failed",
            "details": str(error)
        }), 500


if __name__ == "__main__":
    print("Starting Smart Campus Energy API...")
    app.run(host="127.0.0.1", port=5001, debug=False)
