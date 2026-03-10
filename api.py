from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# 1. Initialize FastAPI app
app = FastAPI(
    title="Telco Churn Prediction API",
    description="API for predicting customer churn using a Random Forest model.",
    version="1.0.0"
)

# 2. Define input data model
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

# 3. Load Model and Scaler at startup
try:
    with open('model_churn_v2.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler_v2.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    model, scaler = None, None

# 4. Endpoints
@app.get("/")
def home():
    return {"message": "Welcome to Telco Churn Prediction API. Go to /docs for interactive documentation."}

@app.post("/predict")
def predict_churn(data: CustomerData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or Scaler not loaded on server.")
    
    try:
        # Prepare input for prediction
        input_array = np.array([[data.tenure, data.MonthlyCharges, data.TotalCharges]])
        
        # Scaling
        input_scaled = scaler.transform(input_array)
        
        # Prediction
        prediction = int(model.predict(input_scaled)[0])
        probability = float(model.predict_proba(input_scaled)[0][1])
        
        return {
            "prediction": "Churn" if prediction == 1 else "Loyal",
            "churn_probability": round(probability, 4),
            "status_code": 200
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
