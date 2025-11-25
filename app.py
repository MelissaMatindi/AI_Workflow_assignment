
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import lightgbm as lgb
import numpy as np

app = FastAPI(title="30-day Readmission Risk API")

# Load artifacts
model = lgb.Booster(model_file='readmission_lightgbm_model.txt')
scaler = joblib.load('scaler.pkl')
features = list(pd.read_csv('feature_names.csv').iloc[:, 0])

class Patient(BaseModel):
    age: int
    gender: str
    los: int
    comorbidities: int
    prior_admissions: int
    admission_type: str
    discharge_disposition: str
    lab_albumin: float

@app.post("/predict")
def predict(patient: Patient):
    # Convert to DataFrame with correct columns
    df = pd.DataFrame([patient.dict()], columns=features)
    
    # Same preprocessing as training (you would copy the function here)
    # For brevity we just scale numerics and one-hot the cats
    num_cols = ['age','los','comorbidities','prior_admissions','lab_albumin']
    cat_cols = ['gender','admission_type','discharge_disposition']
    df = pd.get_dummies(df, columns=cat_cols)
    df[num_cols] = scaler.transform(df[num_cols])
    df = df.reindex(columns=features, fill_value=0)
    
    proba = model.predict(df)[0]
    risk = "High" if proba >= 0.13 else "Low"   # your chosen threshold
    
    return {"readmission_probability": round(float(proba), 3),
            "risk_category": risk}
