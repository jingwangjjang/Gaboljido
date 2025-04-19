import os
from dotenv import load_dotenv
from model.model_ocr.loader import YOLOSubtitleDetector

load_dotenv()

def run_ocr(url: str) -> list:
    detector = YOLOSubtitleDetector(
        model_path="model_ocr/yolo_best.pt", 
        ocr_secret_key=os.getenv("CLOVA_OCR_SECRET_KEY"), 
        ocr_invoke_url=os.getenv("CLOVA_OCR_API_URL")
    )
    return detector.process_youtube_pipeline(url)
