from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import numpy as np
import uvicorn

app = FastAPI(title="Iris Classifier API")

# Charger le modèle
model = load('model.joblib')
scaler = load('scaler.joblib')
target_names = ['setosa', 'versicolor', 'virginica']


class IrisFeatures(BaseModel):
    features: list[float]


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: list[float]


@app.post("/predict", response_model=PredictionResponse)
def predict(iris: IrisFeatures):
    try:
        features = np.array(iris.features).reshape(1, -1)
        if features.shape[1] != 4:
            raise HTTPException(status_code=400, detail="Exactement 4 caractéristiques requises")

        features_scaled = scaler.transform(features)
        prediction = int(model.predict(features_scaled)[0])
        probabilities = model.predict_proba(features_scaled)[0].tolist()

        return {
            "prediction": prediction,
            "class_name": target_names[prediction],
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", reload=True)