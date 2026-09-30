import mlflow.sklearn

from fastapi import FastAPI

app = FastAPI()

# MLflowからモデルを読み込む
model_uri = "mlruns/1/models/m-d2c882540b9041108514a67137e37eff/artifacts"
model = mlflow.sklearn.load_model(model_uri)


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(features: list[float]):
    prediction = model.predict([features])

    return {
        "prediction": int(prediction[0])
    }
