import re
import sys
import threading
import time
import warnings
from io import BytesIO

import joblib
import pandas as pd
import requests
import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor

# Игнорируем всплывающую ошибку о том, что метод уже устарел
warnings.filterwarnings("ignore", category=DeprecationWarning)
DANGEROUS_PATTERN = re.compile(r'^\s*([=+\-@])')

csv_path = "Laptop_price.csv"

def train_model(csv_path, model_path):
    print("Загрузка данных из файла:", csv_path)
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['Price'])
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    num_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X_train.select_dtypes(include=['object']).columns.tolist()

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
    ])
    print("Обучение модели...")
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, model_path)
    print("Модель успешно обучена и сохранена в:", model_path)


app = FastAPI()
model = None
model_path = "laptop_price_model.pkl"

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(model_path)
        print("Модель загружена из", model_path)
    except Exception as e:
        print("Ошибка загрузки модели:", e)
        sys.exit(1)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        preds = model.predict(df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")


if __name__ == "__main__":
    user_choice = input("Введите \"1\" для обучения модели или \"2\" для запуска сервера и запроса: ")

    if user_choice == "1":
        csv_path = "Laptop_price.csv"
        train_model(csv_path, model_path)
    elif user_choice == "2":
        def run_server():
            uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)


        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(2)

        try:
            with open("Laptop_price.csv", "rb") as f:
                files = {"file": f}
                response = requests.post("http://127.0.0.1:8000/predict/", files=files)
                print("Статус-код ответа:", response.status_code)
                print("Ответ сервера:", response.text)
        except FileNotFoundError:
            print("Файл laptop_price_model.pkl не найден в текущей папке.")
        except Exception as e:
            print("Произошла ошибка при запросе:", str(e))
    else:
        print("Неверный выбор. Пожалуйста, введите 1 или 2.")
