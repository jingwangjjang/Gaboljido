from fastapi import FastAPI
from pydantic import BaseModel
from service.pipeline import run_pipeline
from dotenv import load_dotenv
import os
import requests

load_dotenv()
DJANGO_RESULT_API = os.getenv("DJANGO_RESULT_API")

app = FastAPI()

class AnalyzeRequest(BaseModel):
    url: str
    video_id: int
    region_code: int

@app.post("/start-analysis/")
def start_analysis(req: AnalyzeRequest):
    # 🔍 모델 파이프라인 실행
    result = run_pipeline(req.url, req.region_code)

    # 🔄 Django로 결과 전송
    payload = {
        "video_id": req.video_id,
        "url": req.url,
        "result": result
    }

    try:
        res = requests.post(DJANGO_RESULT_API, json=payload)
        res.raise_for_status()
    except Exception as e:
        return {
            "status": "error",
            "code": 500,
            "message": f"Django 저장 실패: {str(e)}",
            "data": result
        }

    return {
        "status": "success",
        "code": 200,
        "message": "모델 분석 완료 및 Django 저장 완료",
        "data": result
    }
