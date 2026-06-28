import pandas as pd
import pandera as pa
from pandera import Column, Check
import os

# Получаем путь к файлу iris.csv относительно текущего скрипта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "iris.csv")


def test_data_schema():
    # Проверяем, существует ли файл
    assert os.path.exists(DATA_PATH), f"Файл данных не найден по пути: {DATA_PATH}"

    # Читаем данные
    df = pd.read_csv(DATA_PATH)

    # Описываем строгий контракт (схему) данных
    schema = pa.DataFrameSchema(
        {
            "sepal length (cm)": Column(float, Check.ge(0)),
            "sepal width (cm)": Column(float, Check.ge(0)),
            "petal length (cm)": Column(float, Check.ge(0)),
            "petal width (cm)": Column(float, Check.ge(0)),
            "target": Column(int, Check.isin([0, 1, 2])),
        }
    )

    # Валидация данных
    schema.validate(df)
