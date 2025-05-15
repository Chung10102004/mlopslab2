# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load vectorizer
try:
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Lỗi khi load vectorizer: {e}")

# Load model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Lỗi khi load model: {e}")

app = FastAPI()

class InputData(BaseModel):
    features: str

@app.get("/")
def read_root():
    return {"message": "ml"}

@app.post("/predict")
def predict(input: InputData):
    try:
        text = input.features
        print("Received input:", text)

        # Kiểm tra định dạng input
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")

        # Vector hóa input
        X = vectorizer.transform([text])
        print("Vectorized input shape:", X.shape)

        # Dự đoán
        prediction = model.predict(X)
        return {"prediction": int(prediction[0])}
    
    except Exception as e:
        print("Lỗi tại server:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
