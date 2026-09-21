from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="Payment Fraud Detection API")

# Load artifacts
model = joblib.load('models/fraud_model.pkl')
scaler = joblib.load('models/scaler.pkl')

class Transaction(BaseModel):
    # Requires input features
    features: list[float]  # Expects 28 PCA features + Time + Amount

@app.post("/predict")
def predict_fraud(data: Transaction):
    raw_input = np.array(data.features).reshape(1, -1)
    
    # Scale Time & Amount (last two elements)
    amount_scaled = scaler.transform(raw_input[:, -1].reshape(-1, 1))
    time_scaled = scaler.transform(raw_input[:, -2].reshape(-1, 1))
    
    processed_input = raw_input.copy()
    processed_input[:, -1] = amount_scaled.flatten()
    processed_input[:, -2] = time_scaled.flatten()
    
    prediction = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0][1]
    
    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability), 4)
    }
