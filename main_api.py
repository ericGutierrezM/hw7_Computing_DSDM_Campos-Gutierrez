from fastapi import FastAPI
import joblib

app = FastAPI()

@app.get("/")
async def root():
    return 'Hello'