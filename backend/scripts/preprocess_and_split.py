# backend/scripts/preprocess_and_split.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

# 폴더 생성
Path("backend/model").mkdir(parents=True, exist_ok=True)
Path("backend/data").mkdir(parents=True, exist_ok=True)

# 데이터 로딩
df = pd.read_csv("data/dummy_customer_data.csv", encoding="utf-8-sig")

# 전처리
df['last_login'] = pd.to_datetime(df['last_login'])
df['days_since_login'] = (pd.Timestamp.today() - df['last_login']).dt.days
df['churned'] = df['payment_status'].map({'paid': 0, 'unpaid': 1})
df = df.drop(columns=['payment_status', 'name', 'email', 'preferred_category', 'last_login'])

# 정규화
scaler = MinMaxScaler()
df[['age', 'watch_time', 'days_since_login']] = scaler.fit_transform(
    df[['age', 'watch_time', 'days_since_login']]
)

# X, y 분리
X = df.drop(columns=['churned'])
y = df['churned']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# 저장
joblib.dump(scaler, "backend/model/scaler.pkl")
X_train.to_csv("backend/data/X_train.csv", index=False)
X_test.to_csv("backend/data/X_test.csv", index=False)
y_train.to_csv("backend/data/y_train.csv", index=False)
y_test.to_csv("backend/data/y_test.csv", index=False)
df.to_csv("data/processed_customer_data.csv", index=False, encoding="utf-8-sig")

print("✅ 전처리 및 분할 완료. scaler 및 데이터 저장됨.")
