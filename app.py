import joblib
import os
import boto3

from fastapi import FastAPI

app = FastAPI()

# S3からモデルをダウンロード
s3 = boto3.client("s3")
model_path = "/tmp/model.pkl"

s3.download_file(
    "mlops-test-models-432214230691",
    "models/iris/model.pkl",
    model_path
)

# ダウンロードしたモデルを読み込む
model = joblib.load(model_path)

@app.get("/")
def root():
    return {
        "message": "ML API is running",
        "hostname": os.environ.get("HOSTNAME")
    }

@app.post("/predict")
def predict(features: list[float]):
    prediction = model.predict([features])
    return {
        "prediction": int(prediction[0])
    }
