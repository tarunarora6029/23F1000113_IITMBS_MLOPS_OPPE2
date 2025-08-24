#!/usr/bin/env python3
"""
FastAPI application for heart disease prediction
Includes logging, monitoring, and observability
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import json
import time
import logging
import structlog
from typing import List, Dict, Any
import os
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
PREDICTION_COUNTER = Counter('heart_disease_predictions_total', 'Total predictions made', ['prediction'])
PREDICTION_LATENCY = Histogram('heart_disease_prediction_duration_seconds', 'Prediction latency')
REQUEST_COUNTER = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="ML model for predicting heart disease with explainability",
    version="1.0.0"
)

# Global variables for model components
model = None
scaler = None
label_encoder = None
feature_names = None
metadata = None

class HeartDiseaseInput(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years")
    gender: int = Field(..., ge=0, le=1, description="Gender (0=male, 1=female)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure")
    chol: float = Field(..., ge=100, le=600, description="Cholesterol level")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results")
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=0, le=2, description="ST segment slope")
    ca: int = Field(..., ge=0, le=3, description="Number of major vessels")
    thal: int = Field(..., ge=1, le=3, description="Thalassemia type")

class PredictionResponse(BaseModel):
    prediction: int
    probability: List[float]
    risk_level: str
    confidence: float
    feature_importance: Dict[str, float]
    timestamp: str
    model_version: str

class BatchPredictionInput(BaseModel):
    samples: List[HeartDiseaseInput]

@app.on_event("startup")
async def load_model():
    """Load model and components on startup"""
    global model, scaler, label_encoder, feature_names, metadata
    
    try:
        logger.info("Loading model components...")
        
        model = joblib.load('models/heart_disease_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        
        with open('models/metadata.json', 'r') as f:
            metadata = json.load(f)
            
        feature_names = metadata['feature_names']
        
        logger.info("Model loaded successfully", 
                   model_type=metadata.get('model_type', 'Unknown'),
                   features=len(feature_names))
        
    except Exception as e:
        logger.error("Failed to load model", error=str(e))
        raise e

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    start_time = time.time()
    
    # Count request
    REQUEST_COUNTER.labels(method=request.method, endpoint=request.url.path).inc()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    logger.info("Request processed",
               method=request.method,
               path=request.url.path,
               status_code=response.status_code,
               process_time=process_time)
    
    return response

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: HeartDiseaseInput):
    """Make a single prediction"""
    
    with PREDICTION_LATENCY.time():
        try:
            # Convert input to DataFrame
            input_dict = input_data.dict()
            df = pd.DataFrame([input_dict])
            
            # Scale features
            X_scaled = scaler.transform(df[feature_names])
            
            # Make prediction
            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0]
            
            # Calculate confidence and risk level
            confidence = max(probabilities)
            risk_level = "High Risk" if prediction == 1 else "Low Risk"
            
            # Get feature importance (coefficients)
            feature_importance = {}
            if hasattr(model, 'coef_'):
                coefficients = model.coef_[0]
                for i, feature in enumerate(feature_names):
                    feature_importance[feature] = float(coefficients[i] * df[feature].iloc[0])
            
            # Count prediction
            PREDICTION_COUNTER.labels(prediction=prediction).inc()
            
            # Log prediction
            logger.info("Prediction made",
                       prediction=int(prediction),
                       probability=probabilities.tolist(),
                       confidence=float(confidence),
                       input_features=input_dict)
            
            response = PredictionResponse(
                prediction=int(prediction),
                probability=probabilities.tolist(),
                risk_level=risk_level,
                confidence=float(confidence),
                feature_importance=feature_importance,
                timestamp=datetime.now().isoformat(),
                model_version="1.0.0"
            )
            
            return response
            
        except Exception as e:
            logger.error("Prediction failed", error=str(e), input=input_data.dict())
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(input_data: BatchPredictionInput):
    """Make batch predictions"""
    
    try:
        predictions = []
        
        for sample in input_data.samples:
            # Reuse single prediction logic
            result = await predict(sample)
            predictions.append(result)
        
        logger.info("Batch prediction completed", batch_size=len(predictions))
        
        return {
            "predictions": predictions,
            "batch_size": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Batch prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_type": metadata.get('model_type', 'Unknown'),
        "features": feature_names,
        "target_classes": metadata.get('target_classes', []),
        "feature_count": len(feature_names),
        "version": "1.0.0"
    }

@app.get("/model/features")
async def get_features():
    """Get feature definitions"""
    feature_descriptions = {
        'age': 'Patient age in years',
        'gender': 'Gender (0=male, 1=female)',
        'cp': 'Chest pain type (0=typical angina, 1=atypical angina, 2=non-anginal pain, 3=asymptomatic)',
        'trestbps': 'Resting blood pressure (mm Hg)',
        'chol': 'Serum cholesterol (mg/dl)',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1=yes, 0=no)',
        'restecg': 'Resting ECG results (0=normal, 1=ST-T wave abnormality, 2=left ventricular hypertrophy)',
        'thalach': 'Maximum heart rate achieved during exercise',
        'exang': 'Exercise induced angina (1=yes, 0=no)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of peak exercise ST segment (0=upsloping, 1=flat, 2=downsloping)',
        'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
        'thal': 'Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)'
    }
    
    return {
        "features": [
            {
                "name": feature,
                "description": feature_descriptions.get(feature, "No description available")
            }
            for feature in feature_names
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
