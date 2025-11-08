import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class PredictionInput(BaseModel):
    features: list

app = FastAPI()

try:
    model = joblib.load('model.pkl')
except Exception as e:
    print('Error loading model', e)
    model = None

@app.post('/predict')
async def predict(item: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail='Model not loaded')
    try:
        prediction = model.predict(np.array(item.features).reshape(1, -1))
        return {'prediction': int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail='Model prediction failed')