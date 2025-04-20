from ocr_utils import download_youtube_video, extract_frames_from_video, preprocess_image_for_ocr
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
import os
import torch
import re
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
    def __init__(self, model_path: str, ocr_secret_key: str, ocr_invoke_url: str, conf_threshold: float = 0.1):
        """
        초기화 함수
        
        Args:
            model_path (str): YOLO 모델 경로 "/model_ocr/yolo_best.pt"
            ocr_secret_key (str): Clova OCR API 키
            ocr_invoke_url (str): Clova OCR API URL
            conf_threshold (float): 검출 신뢰도 임계값으로 줄이면 더 많이 잡는데 오탐이 많아짐 현재 0.3
        """
        self.model_path = "gaboljido_yolo.torchscript"
        self.ocr_secret_key = os.getenv("CLOVA_OCR_SECRET_KEY") or ocr_secret_key
        self.ocr_invoke_url = os.getenv("CLOVA_OCR_API_URL") or ocr_invoke_url
        self.conf_threshold = conf_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ✅ device 저장
        self.model = self._load_model()

    def _load_model(self):
        """YOLO 모델 로드"""
        try:
            logger.info(f"YOLO 모델 로드 중: {self.model_path}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # TorchScript 모델 로드
            model = DetectMultiBackend(self.model_path, device=self.device)
            model.eval()

            logger.info(f"{self.device.type.upper()}에서 모델 로드 완료")
            return model

        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {e}")
            raise RuntimeError(f"모델 로드 실패: {e}")
        
    def detect_subtitle_crops_single(self, images: List[np.ndarray]) -> List[List[np.ndarray]]:
        all_crops = []

        for idx, image in enumerate(images):
            try:
                # 전처리
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                orig_h, orig_w = image.shape[:2]
                resized = cv2.resize(image, (640, 640))
                tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                tensor = tensor.to(self.device)

                # 추론
                with torch.no_grad():
                    preds = self.model(tensor)
                    if isinstance(preds, (list, tuple)):
                        preds = preds[0]
                    if preds.ndim == 2:
                        preds = preds.unsqueeze(0)

                # NMS
                dets = non_max_suppression(preds, conf_thres=self.conf_threshold, iou_thres=0.45)[0]

                # 박스 복원 및 크롭
                crops = []
                if dets is not None and len(dets):
                    for *xyxy, conf, cls in dets:
                        x1, y1, x2, y2 = [int(v.item()) for v in xyxy]

                        # 원본 해상도에 맞게 스케일 조정
                        scale_x = orig_w / 640
                        scale_y = orig_h / 640
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)

                        crop = image[y1:y2, x1:x2]
                        crops.append(crop)

                print(f"[DEBUG] Frame {idx+1}/{len(images)}: 검출된 크롭 수 = {len(crops)}")
                all_crops.append(crops)

            except Exception as e:
                print(f"[ERROR] Frame {idx+1}: {e}")
                all_crops.append([])

        return all_crops

        
    def _call_clova_ocr_image(self, image_np_array, ocr_invoke_url, ocr_secret_key):
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
            'X-OCR-SECRET': ocr_secret_key
        }

        try:
            response = requests.post(ocr_invoke_url, headers=headers, data=payload, files=files, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ CLOVA OCR 요청 실패: {e}")
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
        crops = [crop for batch in crops for crop in batch]
        for crop in crops:
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                texts.append("")
                continue
            # 전처리
            preprocessed = preprocess_image_for_ocr(crop)

            # 클로바 OCR 바로 호출 (NumPy 배열 사용) call_clova_ocr_image는 인식까지만 하고 _extract_text_from_ocr_result는 텍스트 추출출
            ocr_result = self._call_clova_ocr_image(preprocessed, self.ocr_invoke_url, self.ocr_secret_key)
            text = self._extract_text_from_ocr_result(ocr_result)
            texts.append(text)

        return texts
    
    def process_youtube_pipeline(self, youtube_url: str, interval_sec: float = 1.5, region_code: Optional[int] = None) -> List[str]:

        # 1. 유튜브 영상 다운로드
        video_data = download_youtube_video(youtube_url)
        if video_data is None:
            logger.error("❌ YouTube 영상 다운로드 실패")
            return []
        print("youtube download 완료료")

        # 2. 프레임 추출
        frames_info = extract_frames_from_video(video_data["frames"], interval_sec=1.5)
        print(f"🖼️ 총 {len(frames_info)}개 프레임 추출 완료")

        # 3. 자막 검출 + OCR
        raw_texts = []
        for i, frame_info in enumerate(frames_info):
            image = frame_info.get("image")
            if image is None:
                continue

            crops = self.detect_subtitle_crops_single([image])
            print(f"[DEBUG] Frame {i + 1}/{len(frames_info)}: 검출된 크롭 수 = {len(crops)}")
            if len(crops) == 0:
                continue

            texts = self.extract_text_with_ocr(crops)
            for crop_idx, text in enumerate(texts):
                if not text.strip():
                    continue
                logger.info(f"[Frame {i}] Crop {crop_idx}: '{text.strip()}'")
                raw_texts.append(text.strip())
        print("ocr 추출 완료료")

        # ✅ 중복 제거
        deduped_texts = []
        if raw_texts:
            seen = set()
            for text in raw_texts:
                # 한글만 남기고 나머지 제거
                korean_only = re.sub(r'[^가-힣\s]', '', text)
                korean_only = korean_only.strip()
                if korean_only and korean_only not in seen:
                    seen.add(korean_only)
                    deduped_texts.append(korean_only)

        #print(f"최종 반환 text:{deduped_texts}")
        logger.info(f"✅ 파이프라인 완료: {len(deduped_texts)}개 자막 검출됨")
        return deduped_texts


    
    def __del__(self):
        """소멸자: 모델 메모리 해제"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
