import os
from model.model_ocr.loader import YOLOSubtitleDetector

def run_ocr(url: str) -> list:
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 절대경로 기준
    model_path = os.path.join(base_dir, "../model/model_ocr/yolo_best.pt")  # 정확한 경로로 수정
    
    detector = YOLOSubtitleDetector(
        model_path=model_path,
        ocr_secret_key=os.getenv("CLOVA_OCR_SECRET_KEY"),
        ocr_invoke_url=os.getenv("CLOVA_OCR_API_URL")
    )
    return detector.process_youtube_pipeline(url)
