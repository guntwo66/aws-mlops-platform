import mlflow
import mlflow.sklearn
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# データを取得
X, y = load_iris(return_X_y=True)

# 学習用とテスト用に分ける
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# MLflowの実験を指定
mlflow.set_experiment("iris-test")

# MLflowで実験を記録開始
with mlflow.start_run():

    # モデルを作る
    model = LogisticRegression(max_iter=1000)

    # 学習
    model.fit(X_train, y_train)

    joblib.dump(model, "model.pkl")

    # 予測
    y_pred = model.predict(X_test)

    # 精度を計算
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2%}")

    # MLflowに情報を記録
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", accuracy)

    # 学習したモデルもMLflowに保存
    mlflow.sklearn.log_model(
        model,
        name="iris_model" ,
        registered_model_name="iris_model"
    )
