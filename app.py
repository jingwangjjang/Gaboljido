from fastapi import FastAPI
from pydantic import BaseModel
from service.analyze import analyze_video  # 분석 함수 가져옴

from dotenv import load_dotenv
import os


load_dotenv()
DJANGO_RESULT_API = os.getenv("DJANGO_RESULT_API")

app = FastAPI()

# 요청 바디 구조 정의
class AnalyzeRequest(BaseModel):
    video_id: int
    url: str
    title: str

# FastAPI 라우터: /start-analysis/
@app.post("/start-analysis/")
def start_analysis(req: AnalyzeRequest):
    analyze_video(req.video_id, req.url, req.title)
    return {"status": "started", "video_id": req.video_id}
