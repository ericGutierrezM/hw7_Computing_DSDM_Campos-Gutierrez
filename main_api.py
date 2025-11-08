from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import joblib
import json

class Item(BaseModel):
    age: float
    height: float
    weight: float
    aids: int
    cirrhosis: int
    hepatic_failure: int
    immunosuppression: int
    leukemia: int
    lymphoma: int
    solid_tumor_with_metastasis: int

app = FastAPI()

@app.get("/")
async def root():
    return 'Hello'

@app.post("/predict")
async def predict(item: Item):
    try:
        model = joblib.load('model_logistic.pkl')
        input_data = [[item.age, item.height, item.weight, item.aids, item.cirrhosis, item.hepatic_failure, item.immunosuppression, item.leukemia, item.lymphoma, item.solid_tumor_with_metastasis]]
        prediction = model.predict(input_data)
        return {"predicted class": prediction[0].item()}

    except HTTPException:
        return {'message': 'model prediction failed'}
    
@app.post("/predict-file")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        data = json.loads(contents)
        input_data = [[data["age"],data["height"],data["weight"],data["aids"],data["cirrhosis"],data["hepatic_failure"],data["immunosuppression"],data["leukemia"],data["lymphoma"],data["solid_tumor_with_metastasis"]]]
        model = joblib.load('model_logistic.pkl')
        prediction = model.predict(input_data)
        return {"prediction": prediction[0].item()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid file: {e}")