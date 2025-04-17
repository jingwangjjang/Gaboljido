from ocr_utils import download_youtube_video, extract_frames_from_video, preprocess_image_for_ocr
import os
import torch
import numpy as np
import cv2
import logging
import requests
import uuid
import time
import json
import tempfile
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv 
'''
참고
model_path = 
conf_threshold = 0.3 
'''
load_dotenv()
logger = logging.getLogger("subtitle-detector.loader")

class YOLOSubtitleDetector:
    """
    YOLO 모델을 사용한 자막 검출 및 OCR 처리를 수행하는 클래스
    """
    def __init__(self, model_path: str, ocr_secret_key: str, ocr_invoke_url: str, conf_threshold: float = 0.3):
        """
        초기화 함수
        
        Args:
            model_path (str): YOLO 모델 경로 "/model_ocr/yolo_best.pt"
            ocr_secret_key (str): Clova OCR API 키
            ocr_invoke_url (str): Clova OCR API URL
            conf_threshold (float): 검출 신뢰도 임계값으로 줄이면 더 많이 잡는데 오탐이 많아짐 현재 0.3
        """
        self.model_path = model_path
        self.ocr_secret_key = os.getenv("CLOVA_OCR_SECRET_KEY")
        self.ocr_invoke_url = os.getenv("CLOVA_OCR_API_URL")
        self.conf_threshold = conf_threshold
        self.model = self._load_model()
        
    def _load_model(self): 
        """YOLO 모델 로드"""
        try:
            logger.info(f"YOLO 모델 로드 중: {self.model_path}")
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 모델 로드
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload=False)
            model.conf = self.conf_threshold
            
            # GPU 사용 설정
            if torch.cuda.is_available():
                model.cuda()
                logger.info("CUDA 사용 가능: GPU로 모델을 로드했습니다.")
            else:
                logger.info("CUDA 사용 불가: CPU로 모델을 로드했습니다.")
            
            return model
        
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {e}")
            raise RuntimeError(f"모델 로드 실패: {e}")
    

    def detect_subtitle_crops(self, image: np.ndarray) -> list:
        """
        YOLO 모델을 이용하여 단일 이미지에서 자막 영역 검출 후, 해당 영역의 크롭 이미지를 반환

        Args:
            model: YOLO 모델 객체
            image (np.ndarray): 입력 이미지 (BGR 형태), extract_frames_from_video에서 numpy 배열로 변환된 이미지들들

        Returns:
            List[np.ndarray]: 검출된 각 자막 영역에 대한 크롭 이미지 리스트
        """
        try:
            # NumPy 배열을 임시 이미지 파일로 저장
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                cv2.imwrite(temp_file.name, image)
                temp_path = temp_file.name

            try:
                results = self.model(temp_path)
                boxes = results.xyxy[0].cpu().numpy()

                crops = []
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    crop = image[y1:y2, x1:x2]
                    crops.append(crop)
                return crops

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            #print(f"[ERROR] 자막 크롭 중 오류 발생: {e}")
            return []
        
    def _call_clova_ocr_image(self, image_np_array, api_url, secret_key):
        """
        YOLO 모델을 이용하여 단일 이미지에서 자막 영역 검출 후, 해당 영역의 크롭 이미지를 반환

        Args:
            image_np_array : detect_subtitle_crops에서 return한 crop된 이미지들들
            api_url : clova api_url
            secret_key : clova secret key

        Returns:
            
        """
        # numpy array를 JPEG 바이트로 변환
        _, img_encoded = cv2.imencode('.jpg', image_np_array)
        image_bytes = img_encoded.tobytes()

        request_json = {
            'images': [
                {
                    'format': 'jpg',
                    'name': 'box_crop'
                }
            ],
            'requestId': str(uuid.uuid4()),
            'version': 'V2',
            'timestamp': int(round(time.time() * 1000))
        }

        payload = {'message': json.dumps(request_json).encode('UTF-8')}
        files = [('file', ('crop.jpg', image_bytes, 'image/jpeg'))]
        headers = {
            'X-OCR-SECRET': secret_key
        }

        try:
            response = requests.post(api_url, headers=headers, data=payload, files=files, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            #print(f"❌ CLOVA OCR 요청 실패: {e}")
            return {"images": [{"fields": []}]}  # 기본 빈 응답
    
    
    def _extract_text_from_ocr_result(self, result: Dict[str, Any]) -> str:
        """
        OCR 결과에서 텍스트 추출
        
        Args:
            result (Dict[str, Any]): OCR 결과 JSON
            
        Returns:
            str: 추출된 텍스트
        """
        try:
            texts = []
            
            for field in result.get('images', [])[0].get('fields', []):
                text = field.get('inferText', '').strip()
                if text:
                    texts.append(text)
            
            return " ".join(texts)
        
        except Exception as e:
            logger.error(f"OCR 결과 파싱 중 오류 발생: {e}")
            return ""

    def extract_text_with_ocr(self, crops: List[np.ndarray]) -> List[str]:
        """
        여러 개의 크롭된 이미지에 대해 OCR 수행행

        Args:
            crops : numpy 배열의 crop된 이미지들들

        Returns:
            List[str]: 각 이미지에서 추출된 텍스트
        """
        texts = []
        for crop in crops:
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                texts.append("")
                continue
            # 전처리
            preprocessed = preprocess_image_for_ocr(crop)

            # 클로바 OCR 바로 호출 (NumPy 배열 사용) call_clova_ocr_image는 인식까지만 하고 _extract_text_from_ocr_result는 텍스트 추출출
            ocr_result = self._call_clova_ocr_image(preprocessed, self.api_url, self.secret_key)
            text = self._extract_text_from_ocr_result(ocr_result)
            texts.append(text)

        return texts
    
    def process_youtube_pipeline(self, youtube_url: str, interval_sec: float = 1.5) -> List[str]:
        from ocr_utils import download_youtube_video, extract_frames_from_video

        # 1. 유튜브 영상 다운로드
        video_data = download_youtube_video(youtube_url)
        if video_data is None:
            logger.error("❌ YouTube 영상 다운로드 실패")
            return []

        # 2. 프레임 추출
        frames_info = extract_frames_from_video(video_data["frames"], interval_sec)

        # 3. 자막 검출 + OCR
        raw_texts = []
        for i, frame_info in enumerate(frames_info):
            image = frame_info.get("image")
            if image is None:
                continue

            crops = self.detect_subtitle_crops(image)
            if len(crops) == 0:
                continue

            texts = self.extract_text_with_ocr(crops)
            for crop_idx, text in enumerate(texts):
                if not text.strip():
                    continue
                logger.info(f"[Frame {i}] Crop {crop_idx}: '{text.strip()}'")
                raw_texts.append(text.strip())

        # ✅ 중복 제거
        deduped_texts = []
        if raw_texts:
            seen = set()
            for text in raw_texts:
                if text not in seen:
                    seen.add(text)
                    deduped_texts.append(text)

        logger.info(f"✅ 파이프라인 완료: {len(deduped_texts)}개 자막 검출됨")
        return deduped_texts


    
    def __del__(self):
        """소멸자: 모델 메모리 해제"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
