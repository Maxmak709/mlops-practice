import joblib
import os
import pandas as pd


def test_model_exists():
    assert os.path.exists("models/model.pkl")


def test_model_prediction():
    model = joblib.load("models/model.pkl")
    # Тестовый пример (ирис)
    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
    )
    pred = model.predict(sample)
    assert pred[0] in [0, 1, 2]  # Проверка, что класс корректен
