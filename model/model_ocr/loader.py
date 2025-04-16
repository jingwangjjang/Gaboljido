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

logger = logging.getLogger("subtitle-detector.loader")

class YOLOSubtitleDetector:
    """
    YOLO 모델을 사용한 자막 검출 및 OCR 처리를 수행하는 클래스
    """
    def __init__(self, model_path: str, ocr_secret_key: str, ocr_invoke_url: str, conf_threshold: float = 0.25):
        """
        초기화 함수
        
        Args:
            model_path (str): YOLO 모델 경로
            ocr_secret_key (str): Clova OCR API 키
            ocr_invoke_url (str): Clova OCR API URL
            conf_threshold (float): 검출 신뢰도 임계값
        """
        self.model_path = model_path
        self.ocr_secret_key = ocr_secret_key
        self.ocr_invoke_url = ocr_invoke_url
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
    
    def detect_subtitles(self, image: np.ndarray) -> np.ndarray:
        """
        이미지에서 자막 영역 검출
        
        Args:
            image (np.ndarray): 이미지 NumPy 배열
            
        Returns:
            np.ndarray: 검출된 박스 좌표 배열 [x1, y1, x2, y2]
        """
        try:
            # NumPy 이미지를 임시 파일로 저장 (YOLO 모델이 직접 NumPy 배열을 처리할 수 없는 경우)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                cv2.imwrite(temp_file.name, image)
                temp_path = temp_file.name
            
            try:
                # YOLO 추론 실행
                results = self.model(temp_path)
                
                # 결과에서 박스 좌표 추출
                boxes = results.xyxy[0].cpu().numpy()
                
                # 좌표만 반환 (x1, y1, x2, y2)
                return boxes[:, :4]
            
            finally:
                # 임시 파일 삭제
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        except Exception as e:
            logger.error(f"자막 검출 중 오류 발생: {e}")
            return np.array([])
    
    def extract_text_with_ocr(self, image: np.ndarray, boxes: np.ndarray) -> List[str]:
        """
        이미지에서 검출된 박스 영역의 텍스트 추출
        
        Args:
            image (np.ndarray): 이미지 NumPy 배열
            boxes (np.ndarray): 검출된 박스 좌표 배열
            
        Returns:
            List[str]: 각 박스별 추출된 텍스트 목록
        """
        try:
            texts = []
            
            # 각 박스 영역에 대해 OCR 수행
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                
                # 박스 영역 추출
                cropped = image[y1:y2, x1:x2]
                
                # 너무 작은 영역은 제외
                if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
                    texts.append("")
                    continue
                
                # OCR 수행
                ocr_result = self._call_clova_ocr(cropped)
                text = self._extract_text_from_ocr_result(ocr_result)
                texts.append(text)
            
            return texts
        
        except Exception as e:
            logger.error(f"OCR 처리 중 오류 발생: {e}")
            return []
    
    def _call_clova_ocr(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Clova OCR API 호출
        
        Args:
            image (np.ndarray): OCR을 수행할 이미지
            
        Returns:
            Dict[str, Any]: OCR 결과 JSON
        """
        try:
            # 이미지를 JPEG 바이트로 변환
            _, img_encoded = cv2.imencode('.jpg', image)
            image_bytes = img_encoded.tobytes()
            
            # API 요청 데이터 준비
            request_json = {
                'images': [
                    {
                        'format': 'jpg',
                        'name': 'subtitle_area'
                    }
                ],
                'requestId': str(uuid.uuid4()),
                'version': 'V2',
                'timestamp': int(round(time.time() * 1000))
            }
            
            payload = {'message': json.dumps(request_json).encode('UTF-8')}
            files = [('file', ('image.jpg', image_bytes, 'image/jpeg'))]
            headers = {
                'X-OCR-SECRET': self.ocr_secret_key
            }
            
            # API 호출
            response = requests.post(
                self.ocr_invoke_url,
                headers=headers,
                data=payload,
                files=files,
                timeout=10
            )
            response.raise_for_status()
            
            return response.json()
        
        except Exception as e:
            logger.error(f"Clova OCR API 호출 중 오류 발생: {e}")
            return {"images": [{"fields": []}]}
    
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
    
    def process_frames(self, frames_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        프레임 이미지들에 대해 자막 검출 및 OCR 처리 수행
        
        Args:
            frames_info (List[Dict[str, Any]]): 프레임 정보 목록
                {
                    "frame_number": int,
                    "timestamp": float,
                    "image": np.ndarray
                }
            
        Returns:
            List[Dict[str, Any]]: 자막 검출 결과 목록
        """
        results = []
        total_frames = len(frames_info)
        
        logger.info(f"총 {total_frames}개 프레임 처리 시작")
        
        for i, frame_info in enumerate(frames_info):
            # 프레임 번호와 타임스탬프 가져오기
            frame_number = frame_info.get("frame_number", i)
            timestamp = frame_info.get("timestamp", 0)
            
            # 이미지 데이터 가져오기
            image = frame_info.get("image")
            if image is None:
                logger.warning(f"프레임 {i}에 이미지 데이터가 없습니다.")
                continue
            
            # 처리 진행 상황 로깅
            if (i + 1) % 10 == 0 or i + 1 == total_frames:
                logger.info(f"프레임 처리 중: {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%)")
            
            # 자막 영역 검출
            boxes = self.detect_subtitles(image)
            
            # 검출된 자막 영역이 없으면 다음 프레임으로
            if len(boxes) == 0:
                continue
            
            # OCR 수행
            texts = self.extract_text_with_ocr(image, boxes)
            
            # 결과 저장
            for box_idx, (box, text) in enumerate(zip(boxes, texts)):
                if not text:  # 빈 텍스트는 건너뜀
                    continue
                
                results.append({
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "box_index": box_idx,
                    "box_coordinates": box.tolist(),
                    "text": text
                })
        
        logger.info(f"프레임 처리 완료: {len(results)}개 자막 검출됨")
        return results
    
    def process_single_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        단일 이미지에 대해 자막 검출 및 OCR 처리 수행
        
        Args:
            image (np.ndarray): 이미지 NumPy 배열
            
        Returns:
            List[Dict[str, Any]]: 자막 검출 결과 목록
        """
        try:
            # 자막 영역 검출
            boxes = self.detect_subtitles(image)
            
            # 검출된 자막 영역이 없으면 빈 리스트 반환
            if len(boxes) == 0:
                return []
            
            # OCR 수행
            texts = self.extract_text_with_ocr(image, boxes)
            
            # 결과 생성
            results = []
            for box_idx, (box, text) in enumerate(zip(boxes, texts)):
                if not text:  # 빈 텍스트는 건너뜀
                    continue
                
                results.append({
                    "box_index": box_idx,
                    "box_coordinates": box.tolist(),
                    "text": text
                })
            
            return results
        
        except Exception as e:
            logger.error(f"이미지 처리 중 오류 발생: {e}")
            return []
    
    def process_video(self, video_data: np.ndarray, interval_sec: float = 1.5) -> List[Dict[str, Any]]:
        """
        비디오 NumPy 배열을 처리하여 자막을 검출합니다.
        
        Args:
            video_data (np.ndarray): 비디오 데이터 NumPy 배열 (프레임, 높이, 너비, 채널)
            interval_sec (float): 프레임 추출 간격(초)
            
        Returns:
            List[Dict[str, Any]]: 자막 검출 결과 목록
        """
        from model_ocr.utils import extract_frames_from_video
        
        # 프레임 추출
        frames_info = extract_frames_from_video(video_data, interval_sec)
        
        # 추출된 프레임에서 자막 검출 및 OCR 수행
        return self.process_frames(frames_info)
    
    def process_video_file(self, file_path: str, interval_sec: float = 1.5) -> List[Dict[str, Any]]:
        """
        비디오 파일을 처리하여 자막을 검출합니다.
        
        Args:
            file_path (str): 비디오 파일 경로
            interval_sec (float): 프레임 추출 간격(초)
            
        Returns:
            List[Dict[str, Any]]: 자막 검출 결과 목록
        """
        from model_ocr.utils import extract_frames_from_file
        
        # 프레임 추출
        frames_info = extract_frames_from_file(file_path, interval_sec)
        
        # 추출된 프레임에서 자막 검출 및 OCR 수행
        return self.process_frames(frames_info)
    
    def process_youtube_url(self, youtube_url: str, interval_sec: float = 1.5) -> List[Dict[str, Any]]:
        """
        YouTube URL에서 영상을 다운로드하고 자막을 검출합니다.
        
        Args:
            youtube_url (str): YouTube URL
            interval_sec (float): 프레임 추출 간격(초)
            
        Returns:
            List[Dict[str, Any]]: 자막 검출 결과 목록
        """
        from model_ocr.utils import download_youtube_video
        
        # YouTube 영상 다운로드
        video_data = download_youtube_video(youtube_url)
        
        if video_data is None:
            logger.error("YouTube 영상 다운로드 실패")
            return []
        
        # 비디오 처리
        return self.process_video(video_data, interval_sec)
    
    def __del__(self):
        """소멸자: 모델 메모리 해제"""
        if hasattr(self, 'model'):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()