import mlflow.pyfunc
import pandas as pd
import os

# Путь к БД MLflow
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "..", "mlflow.db")
sqlite_uri = f"sqlite:///{db_path}"

mlflow.set_tracking_uri(sqlite_uri)

# Указываем имя модели и алиас
model_name = "Iris_RF_Model"
alias = "champion"

# URI модели с использованием алиаса
model_uri = f"models:/{model_name}@{alias}"
print(f"Скачиваем модель: {model_uri}")

# Загрузка модели из реестра
model = mlflow.pyfunc.load_model(model_uri)

# Тестовые данные (цветок ириса)
dummy_data = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ],
)

# Делаем предсказание
prediction = model.predict(dummy_data)
print(f"\n Предсказание боевой модели: Класс {prediction[0]}")
