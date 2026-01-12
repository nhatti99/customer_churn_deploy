from re import S
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import os
from pathlib import Path

app = FastAPI()

#Load model and scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), "bank_churn_ann.h5")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

class CustomerFeatures(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

geo_map = {'France':0, 'Spain':1, 'Germany':2}
gender_map = {'Female':0, 'Male':1}

@app.post('/predict')
def predict_churn(features: CustomerFeatures):
    data = features.dict()
    #Encode Categorical
    data['Geography'] = geo_map.get(data['Geography'], 0)
    data['Gender'] = gender_map.get(data['Gender'], 0)
    #Convert to DataFrame
    X = pd.DataFrame([data])
    #Scale
    X_scaled = scaler.transform(X)
    #Predict
    pred = model.predict(X_scaled)[0][0]
    return {'churn_probability': float(pred), 'churned':int(pred > 0.5)}
