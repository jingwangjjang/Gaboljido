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
    region_code: int  # ✅ 추가

@app.post("/start-analysis/")
def start_analysis(req: AnalyzeRequest):
    result = run_pipeline(req.url, req.region_code)

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
            "message": str(e),
            "data": result  # FastAPI 자체 응답에는 분석 결과 포함
        }

    return {
        "status": "success",
        "code": 200,
        "message": "모델 분석 완료",
        "data": result  # ✅ RAG 결과 그대로 반환
    }
