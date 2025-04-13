import os
import logging
import requests
from dotenv import load_dotenv
from requests.exceptions import RequestException, Timeout, ConnectionError

# ========== 환경 설정 ==========
load_dotenv()
DJANGO_RESULT_API = os.getenv("DJANGO_RESULT_API")

# ========== 로깅 설정 ==========
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ========== 전역 모델 불러오기  ==========
# 예시:
# from transformers import pipeline
# summarizer = pipeline("summarization", model="your-org/your-summary-model")
# classifier = pipeline("text-classification", model="your-org/your-classifier")

# ========== 분석 파이프라인 시작 ==========

def run_stt(url: str) -> str:
    """
    [모델1] 영상 URL로부터 텍스트 추출 (예: Whisper 등)
    - 입력: 영상 URL
    - 출력: 텍스트로 변환된 스크립트 (string)
    """

    return "텍스트로 변환된 스크립트 (string)"


def summarize(text: str) -> str:
    """
    [모델2] 텍스트 요약 (예: Hugging Face summarization 모델)
    - 입력: 전체 스크립트
    - 출력: 요약된 문장 (string)
    """
  
    return "요약된 문장 (string)"


def extract_keywords(text: str) -> list:
    """
    [모델3] 키워드 추출 (예: KeyBERT, NER 등)
    - 입력: 요약 텍스트
    - 출력: 주요 키워드 리스트 (list of str)
    """

    return ["주요 키워드 리스트"]


def classify(text: str) -> tuple[str, str]:
    """
    [모델4] 감정 분석 및 카테고리 분류 (예: text-classification)
    - 입력: 요약 텍스트
    - 출력: 감정 레이블 (positive/neutral/negative), 카테고리명
    """
    # TODO: 감정 분석 및 카테고리 분류 모델 사용
    return "positive"


# ========== 메인 함수 ==========
# video_id: int, url: str, region: str 3가지가 인자로 들어갈 예정

def analyze_video(video_id: int, url: str, region: str):
    """
    백엔드에서 받은 분석 요청 처리 함수
    """

    try:
        # Step 1: STT (영상 → 텍스트)
        transcript = run_stt(url)

        # Step 2: 요약
        summary = summarize(transcript)

        # Step 3: 키워드 추출
        keywords = extract_keywords(summary)

        # Step 4: 감정 분석 & 분류
        sentiment, category = classify(summary)

        # 영상 길이 등 추가 정보 (지금은 예시로 180초)
        duration = 180

        # 최종 결과 추후 수정 예정
        result = {
            "video_id": video_id,
            "summary": summary,
            "category": category,
            "duration": duration,
            "status": "completed"
        }

        # ========== Django로 결과 전송 ==========
        response = requests.post(DJANGO_RESULT_API, json=result, timeout=5)
        if response.status_code == 200:
            logger.info(f"결과 전송 성공 (status={response.status_code})")
        else:
            logger.warning(f"응답 이상 (status={response.status_code})")

    except (ConnectionError, Timeout) as net_err:
        logger.error(f"Django 연결 실패: {net_err}")
    except RequestException as req_err:
        logger.error(f"Django 요청 실패: {req_err}")
    except Exception as e:
        logger.exception(f"예기치 못한 파이프라인 오류 발생: {e}")
