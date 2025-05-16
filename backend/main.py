from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
from pathlib import Path
import shutil
import json
from fastapi.responses import FileResponse

from scripts.predict_with_model import predict_churn

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# 1. 업로드 → 예측 실행 및 stats 반환
@app.post("/upload")
async def upload(file: UploadFile, threshold: float = Form(...)):
    file_path = Path("backend/data/uploaded.csv")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 예측 실행
    result_df, high_risk_df, stats = predict_churn(file_path, threshold)

    # stats.json 저장 (프론트가 불러올 수 있도록 public 폴더에)
    stats_path = Path("public/stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("📊 최종 통계 결과:", stats)
    return {"stats": stats}

# 2. stats.json 읽기용 GET API (프론트에서 자동 fetch용)
@app.get("/stats")
async def get_stats():
    stats_path = Path("public/stats.json")
    if not stats_path.exists():
        return JSONResponse(content={"error": "stats.json not found"}, status_code=404)

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    return stats


@app.get("/download")
def download_excel():
    file_path = "backend/high_risk_customers.xlsx"
    return FileResponse(
        path=file_path,
        filename="high_risk_customers.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
@app.get("/download")
def download_excel():
    file_path = "backend/high_risk_customers.xlsx"
    return FileResponse(path=file_path, filename="high_risk_customers.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@app.get("/high-risk-customers")
async def get_high_risk_customers():
    df = pd.read_excel("backend/high_risk_customers.xlsx")
    return JSONResponse(content=df.to_dict(orient="records"))