import re
import warnings
from io import BytesIO

import joblib
import pandas as pd

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


model = None
model_path = "laptop_price_model.pkl"

if __name__ == "__main__":
    user_choice = input("Введите \"1\" для обучения модели: ")

    if user_choice == "1":
        csv_path = "Laptop_price.csv"
        train_model(csv_path, model_path)
    else:
        print("Неверный выбор.")
