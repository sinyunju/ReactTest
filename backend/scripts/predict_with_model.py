import pandas as pd
import joblib
from pathlib import Path
from collections import Counter
import json

def predict_churn(file_path: str, threshold: float = 0.5):
    # 1. 데이터 로드
    df = pd.read_csv(file_path)
    df['last_login'] = pd.to_datetime(df['last_login'])
    df['days_since_login'] = (pd.Timestamp.today() - df['last_login']).dt.days

    # 신원 정보 백업
    identity_df = df[['name', 'email', 'age', 'preferred_category']].copy()

    # 2. 머신러닝 입력용 컬럼 선택
    X = df.drop(columns=['name', 'email', 'preferred_category', 'last_login'])

    # 3. 정규화 및 예측
    scaler = joblib.load("backend/model/scaler.pkl")
    model = joblib.load("backend/model/model.pkl")
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]

    # 4. 결과 결합
    result_df = identity_df.copy()
    result_df['churn_probability'] = probs
    result_df['is_high_risk'] = result_df['churn_probability'] > threshold

    # 5. 그룹별 분리
    churn_group = result_df[result_df["is_high_risk"] == True]
    non_churn_group = result_df[result_df["is_high_risk"] == False]

    # 6. 엑셀로 저장
    high_risk_path = Path("backend/high_risk_customers.xlsx")
    churn_group.to_excel(high_risk_path, index=False)

    # 7. 통계 데이터 생성
    stats = {
        "total_customers": len(result_df),
        "expected_churns": int(result_df["is_high_risk"].sum()),

        "average_age": {
            "churn": round(churn_group["age"].mean(), 2) if not churn_group.empty else 0,
            "non_churn": round(non_churn_group["age"].mean(), 2) if not non_churn_group.empty else 0,
        },
        "average_watch_time": {
            "churn": round(df.loc[result_df["is_high_risk"], "watch_time"].mean(), 2) if not churn_group.empty else 0,
            "non_churn": round(df.loc[~result_df["is_high_risk"], "watch_time"].mean(), 2) if not non_churn_group.empty else 0,
        },
        "average_days_since_login": {
            "churn": round(df.loc[result_df["is_high_risk"], "days_since_login"].mean(), 2) if not churn_group.empty else 0,
            "non_churn": round(df.loc[~result_df["is_high_risk"], "days_since_login"].mean(), 2) if not non_churn_group.empty else 0,
        },
        "genre_distribution": dict(Counter(result_df["preferred_category"]))
    }
    # 8. stats.json 저장 (ReactTest-main/public)
    stats_path = Path("../public/stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("✅ 예측 완료: 결과 stats 생성 및 엑셀 저장됨")
    return result_df, churn_group, stats



if __name__ == "__main__":
    result_df, high_risk_df, stats = predict_churn(
        file_path="data/dummy_customer_data_no_payment.csv",
        threshold=0.5
    )
    print(stats)