import os
import yaml
import joblib
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Убедитесь, что пути корректны
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train_model():
    # Читаем параметры
    with open(os.path.join(BASE_DIR, "..", "params.yaml"), "r") as f:
        params = yaml.safe_load(f)["train"]

    # Читаем данные
    df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "iris.csv"))
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Устанавливаем URI для БД и эксперимента
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Iris_Classification")

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Логируем параметры и метрики
        mlflow.log_params(params)
        acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", acc)

        # Явное логирование модели
        mlflow.sklearn.log_model(model, "model")

        print(f"Модель обучена. Accuracy: {acc}")

    # Сохраняем для DVC (вне MLflow)
    metrics_path = os.path.join(BASE_DIR, "..", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc}, f)

    os.makedirs(os.path.join(BASE_DIR, "..", "models"), exist_ok=True)
    joblib.dump(model, os.path.join(BASE_DIR, "..", "models", "model.pkl"))


if __name__ == "__main__":
    train_model()
