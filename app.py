
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
import uvicorn

# Load the saved model and preprocessor
try:
    model = joblib.load('models/diabetes_gradient_boosting_model.pkl')
    preprocessor = joblib.load('models/diabetes_preprocessor.pkl')
    
    # Load metadata for model info
    import json
    with open('models/model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
        
except FileNotFoundError:
    raise Exception("Model files not found. Please ensure models are saved first.")

app = FastAPI(
    title="Diabetes Prediction API",
    description="API for predicting diabetes risk using patient health data",
    version="1.0.0"
)

# Input data model
class PatientData(BaseModel):
    gender: str = Field(..., description="Gender: Male, Female, or Other")
    age: float = Field(..., ge=0, le=120, description="Age in years (0-120)")
    hypertension: int = Field(..., ge=0, le=1, description="Hypertension: 0=No, 1=Yes")
    heart_disease: int = Field(..., ge=0, le=1, description="Heart disease: 0=No, 1=Yes")
    smoking_history: str = Field(..., description="Smoking history: never, former, current, not current, ever, No Info")
    bmi: float = Field(..., ge=10, le=100, description="BMI (10-100)")
    HbA1c_level: float = Field(..., ge=3, le=15, description="HbA1c level (3-15%)")
    blood_glucose_level: int = Field(..., ge=50, le=500, description="Blood glucose level (50-500 mg/dL)")

    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "age": 45.0,
                "hypertension": 0,
                "heart_disease": 0,
                "smoking_history": "never",
                "bmi": 28.5,
                "HbA1c_level": 6.2,
                "blood_glucose_level": 140
            }
        }

# Response model
class PredictionResponse(BaseModel):
    diabetes_probability: float
    diabetes_prediction: str
    risk_level: str
    confidence: str
    model_version: str

@app.get("/")
async def root():
    return {
        "message": "Diabetes Prediction API",
        "model_performance": {
            "accuracy": model_metadata.get("test_accuracy"),
            "recall": model_metadata.get("test_recall"),
            "precision": model_metadata.get("test_precision"),
            "f1_score": model_metadata.get("test_f1")
        }
    }

@app.get("/model-info")
async def get_model_info():
    return model_metadata

@app.post("/predict", response_model=PredictionResponse)
async def predict_diabetes(patient: PatientData):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([patient.dict()])
        
        # Preprocess the data
        processed_data = preprocessor.transform(input_data)
        
        # Make prediction
        probability = model.predict_proba(processed_data)[0][1]
        
        # Apply optimal threshold
        optimal_threshold = model_metadata.get("optimal_threshold", 0.5)
        prediction = "Diabetic" if probability >= optimal_threshold else "Non-Diabetic"
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low Risk"
            confidence = "High"
        elif probability < 0.7:
            risk_level = "Moderate Risk"
            confidence = "Medium"
        else:
            risk_level = "High Risk"
            confidence = "High"
        
        return PredictionResponse(
            diabetes_probability=round(probability, 4),
            diabetes_prediction=prediction,
            risk_level=risk_level,
            confidence=confidence,
            model_version=model_metadata.get("model_type", "Unknown")
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
