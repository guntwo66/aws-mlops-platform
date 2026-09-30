import mlflow.sklearn

model_uri = "runs:/286657844f67409297b0aeda4e2d2404/iris_model"

model = mlflow.sklearn.load_model(model_uri)

print("Model loaded successfully")

# 花の特徴量
sample = [[5.1, 3.5, 1.4, 0.2]]

# 予測
prediction = model.predict(sample)

print(f"Prediction: {prediction}")
