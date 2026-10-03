import joblib

from fastapi import FastAPI

app = FastAPI()

# MLflowからモデルを読み込む
model = joblib.load("model.pkl")

@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(features: list[float]):
    prediction = model.predict([features])

    return {
        "prediction": int(prediction[0])
    }
