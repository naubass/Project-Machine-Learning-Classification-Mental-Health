from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model dan Features
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "../model_rf.pkl")
features_path = os.path.join(BASE_DIR, "../features.pkl")

model = pickle.load(open(model_path, 'rb'))
features = pickle.load(open(features_path, 'rb'))

class UserInput(BaseModel):
    sleep_hours: float
    stress_level: float
    daily_social_media_hours: float
    anxiety_level: float
    gender: int
    platform_usage: int
    social_interaction_level: int
    academic_performance: int
    screen_time_before_sleep: int
    physical_activity: int
    age: int
    addiction_level: int

# Endpoint API untuk Prediksi
@app.post("/predict")
def predict(data: UserInput):
    df = pd.DataFrame([data.dict()])
    df = df[features]
    
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return {
        "prediction": "Depression" if prediction == 1 else "No Depression",
        "confidence": round(float(probability) * 100, 2)
    }

# Melayani File HTML (Frontend)
@app.get("/")
def index():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)