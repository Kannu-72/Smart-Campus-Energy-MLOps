from app import app


def test_home_endpoint():
    """Check whether the Flask application is running."""
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "ok"
    assert data["service"] == "smart-campus-energy-prediction"


def test_prediction_rejects_missing_features():
    """Check whether incomplete prediction input is rejected."""
    client = app.test_client()

    response = client.post("/predict", json={})

    assert response.status_code == 400