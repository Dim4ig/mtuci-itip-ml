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
from cryptography.fernet import Fernet
from fastapi import FastAPI, File, UploadFile, HTTPException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor

# Игнорируем всплывающую ошибку о том, что метод уже устарел
warnings.filterwarnings("ignore", category=DeprecationWarning)
FERNET_KEY = b'tAmMq9m1OD2BP2Cq0oHGrhRWLnujQAGwl4tAqTgE5kk='
DANGEROUS_PATTERN = re.compile(r'^\s*([=+\-@])')


def check_csv_injection(cell):
    if isinstance(cell, str) and DANGEROUS_PATTERN.match(cell):
        return True
    return False


def check_sql_injection(cell):
    if isinstance(cell, str) and '--' in cell:
        return True
    return False


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

        def sanitize_value(val):
            if isinstance(val, str):
                if check_csv_injection(val):
                    raise ValueError(f"Обнаружена CSV-инъекция в столбце {col}: {val}")
                if check_sql_injection(val):
                    raise ValueError(f"Обнаружена SQL-инъекция в столбце {col}: {val}")
            return val

        df[col] = df[col].apply(sanitize_value)
    return df


fernet = Fernet(FERNET_KEY)


def encrypt_ram(df: pd.DataFrame, ram_column: str = "RAM_Size", new_column: str = "RAM_encrypted") -> pd.DataFrame:
    if ram_column in df.columns:
        df[new_column] = df[ram_column].apply(lambda x: fernet.encrypt(str(x).encode()).decode())
    return df


def decrypt_ram_values(encrypted_values):
    decrypted = []
    for value in encrypted_values:
        try:
            decrypted.append(fernet.decrypt(value.encode()).decode())
        except Exception:
            decrypted.append("Ошибка расшифровки")
    return decrypted


def print_first_five_decrypted_ram(df: pd.DataFrame, encrypted_column: str = "RAM_encrypted") -> None:
    if encrypted_column in df.columns:
        sample_values = df[encrypted_column].dropna().head(5).tolist()
        decrypted = decrypt_ram_values(sample_values)
        print("Первые 5 расшифрованных значений RAM:")
        for val in decrypted:
            print(val)


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

        try:
            df = sanitize_dataframe(df)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f"Проверка безопасности не пройдена: {str(ve)}")

        df_encrypted = encrypt_ram(df.copy(), ram_column="RAM_Size", new_column="RAM_encrypted")
        print_first_five_decrypted_ram(df_encrypted, encrypted_column="RAM_encrypted")
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
