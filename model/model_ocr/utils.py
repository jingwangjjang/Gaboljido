import os
import re
import cv2
import subprocess
import logging
import tempfile
import numpy as np
import uuid
from typing import List, Dict, Any, Optional, Tuple, BinaryIO

logger = logging.getLogger("subtitle-detector.utils")

def download_youtube_video(url: str) -> Optional[np.ndarray]:
    """
    yt-dlp를 사용하여 YouTube 영상을 다운로드하고 NumPy 배열로 반환합니다.
    
    Args:
        url (str): YouTube URL
    
    Returns:
        Optional[np.ndarray]: 비디오 데이터의 NumPy 배열, 실패 시 None
    """
    try:
        # YouTube URL 형식 검증 및 변환
        url = convert_youtube_url(url)
        if not url:
            logger.error("유효하지 않은 YouTube URL입니다.")
            return None
        
        logger.info(f"YouTube 영상 다운로드 중: {url}")
        
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # yt-dlp 명령 구성
            cmd = [
                "yt-dlp",
                url,
                "-f", "best[ext=mp4]",  # 최상의 mp4 포맷
                "-o", temp_path,        # 임시 출력 파일 경로
                "--no-playlist",        # 재생목록이 아닌 단일 영상만 다운로드
                "--quiet"               # 상세 출력 비활성화
            ]
            
            # 명령 실행
            subprocess.run(cmd, check=True)
            
            # 다운로드 결과 확인
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                logger.info(f"YouTube 영상 다운로드 완료: {temp_path}")
                
                # 비디오 파일을 NumPy 배열로 변환
                cap = cv2.VideoCapture(temp_path)
                if not cap.isOpened():
                    logger.error("다운로드한 비디오 파일을 열 수 없습니다.")
                    return None
                
                # 비디오 속성 가져오기
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                logger.info(f"비디오 정보: {fps:.2f} FPS, {width}x{height}, 총 {total_frames}프레임")
                
                # 비디오 데이터를 NumPy 배열로 읽기
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                
                cap.release()
                
                # 3차원 배열 (프레임, 높이, 너비, 채널)로 변환
                return np.array(frames)
            else:
                logger.error("다운로드된 파일이 없거나 크기가 0입니다.")
                return None
                
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"임시 파일 삭제됨: {temp_path}")
    
    except subprocess.CalledProcessError as e:
        logger.error(f"yt-dlp 실행 중 오류 발생: {e}")
        return None
    except Exception as e:
        logger.error(f"YouTube 다운로드 중 오류 발생: {e}")
        return None

def convert_youtube_url(url: str) -> Optional[str]:
    """
    다양한 형식의 YouTube URL을 표준 형식으로 변환합니다.
    
    Args:
        url (str): 변환할 YouTube URL
    
    Returns:
        Optional[str]: 변환된 URL 또는 유효하지 않은 경우 None
    """
    # 유효한 YouTube URL 패턴
    youtube_patterns = [
        # 일반 watch URL
        r'(https?://)?(www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        # 단축 URL
        r'(https?://)?(www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
        # Shorts URL
        r'(https?://)?(www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})'
    ]
    
    for pattern in youtube_patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(3)
            return f"https://www.youtube.com/watch?v={video_id}"
    
    return None

def extract_frames_from_video(video_data: np.ndarray, interval_sec: float = 1.5) -> List[Dict[str, Any]]:
    """
    비디오 NumPy 배열에서 정해진 간격으로 프레임을 추출합니다.
    
    Args:
        video_data (np.ndarray): 비디오 데이터 NumPy 배열 (프레임, 높이, 너비, 채널)
        interval_sec (float): 프레임 추출 간격(초)
    
    Returns:
        List[Dict[str, Any]]: 추출된 프레임 정보 목록
    """
    if video_data is None or len(video_data) == 0:
        logger.error("유효한 비디오 데이터가 없습니다.")
        return []
    
    try:
        # 임시 파일에 비디오 저장 (OpenCV에서 fps 정보를 얻기 위함)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # 비디오 쓰기 객체 생성
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width = video_data[0].shape[:2]
            fps = 30.0  # 기본값 (정확한 fps 정보가 없으므로)
            
            out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
            
            # 모든 프레임 쓰기
            for frame in video_data:
                out.write(frame)
            
            out.release()
            
            # 비디오 속성 가져오기
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                logger.error("임시 비디오 파일을 열 수 없습니다.")
                # fps 정보를 얻을 수 없으므로 기본값 사용
                fps = 30.0
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
            
            # 총 프레임 수
            total_frames = len(video_data)
            duration = total_frames / fps
            
            logger.info(f"비디오 정보: {duration:.2f}초, {fps:.2f} FPS, 총 {total_frames}프레임")
            
            # 프레임 추출 간격 계산 (프레임 단위)
            frame_interval = int(fps * interval_sec)
            if frame_interval <= 0:
                frame_interval = 1  # 최소 1 프레임 간격
                
            expected_frames = total_frames // frame_interval + 1
            
            logger.info(f"프레임 추출 간격: {interval_sec}초 ({frame_interval}프레임마다), 예상 추출 프레임: 약 {expected_frames}개")
            
            # 프레임 추출
            frames_info = []
            
            for frame_idx in range(0, total_frames, frame_interval):
                if frame_idx >= total_frames:
                    break
                    
                # 시간 계산
                timestamp = frame_idx / fps
                
                # 프레임 정보 저장
                frames_info.append({
                    "frame_number": frame_idx,
                    "timestamp": timestamp,
                    "image": video_data[frame_idx]
                })
                
                # 진행 상황 로깅
                if len(frames_info) % 10 == 0 or len(frames_info) == expected_frames:
                    logger.info(f"프레임 추출: {len(frames_info)}/{expected_frames} ({len(frames_info)/expected_frames*100:.1f}%)")
            
            logger.info(f"프레임 추출 완료: 총 {len(frames_info)}개 프레임 추출됨")
            return frames_info
            
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        logger.error(f"프레임 추출 중 오류 발생: {e}")
        return []

def extract_frames_from_file(file_path: str, interval_sec: float = 1.5) -> List[Dict[str, Any]]:
    """
    비디오 파일에서 정해진 간격으로 프레임을 추출하여 NumPy 배열로 반환합니다.
    
    Args:
        file_path (str): 비디오 파일 경로
        interval_sec (float): 프레임 추출 간격(초)
    
    Returns:
        List[Dict[str, Any]]: 추출된 프레임 정보 목록, 각 항목은 NumPy 배열 이미지 포함
    """
    try:
        # 비디오 파일 열기
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logger.error(f"비디오 파일을 열 수 없음: {file_path}")
            return []
        
        # 비디오 속성 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"비디오 정보: {duration:.2f}초, {fps:.2f} FPS, 총 {total_frames}프레임")
        
        # 프레임 추출 간격 계산 (프레임 단위)
        frame_interval = int(fps * interval_sec)
        if frame_interval <= 0:
            frame_interval = 1  # 최소 1 프레임 간격
            
        expected_frames = total_frames // frame_interval + 1
        
        logger.info(f"프레임 추출 간격: {interval_sec}초 ({frame_interval}프레임마다), 예상 추출 프레임: 약 {expected_frames}개")
        
        # 프레임 추출
        frame_count = 0
        saved_count = 0
        frames_info = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 지정된 간격마다 프레임 저장
            if frame_count % frame_interval == 0:
                # 프레임 정보 저장 (NumPy 배열 포함)
                frames_info.append({
                    "frame_number": frame_count,
                    "timestamp": frame_count / fps,
                    "image": frame  # NumPy 배열 직접 저장
                })
                
                saved_count += 1
                
                # 진행 상황 로깅
                if saved_count % 10 == 0 or saved_count == expected_frames:
                    logger.info(f"프레임 추출: {saved_count}/{expected_frames} ({saved_count/expected_frames*100:.1f}%)")
            
            frame_count += 1
        
        # 비디오 파일 닫기
        cap.release()
        
        logger.info(f"프레임 추출 완료: 총 {saved_count}개 프레임 추출됨")
        return frames_info
    
    except Exception as e:
        logger.error(f"프레임 추출 중 오류 발생: {e}")
        return []

def extract_frames_from_buffer(video_buffer: bytes, interval_sec: float = 1.5) -> List[Dict[str, Any]]:
    """
    비디오 바이트 버퍼에서 정해진 간격으로 프레임을 추출하여 NumPy 배열로 반환합니다.
    
    Args:
        video_buffer (bytes): 비디오 데이터 바이트 버퍼
        interval_sec (float): 프레임 추출 간격(초)
    
    Returns:
        List[Dict[str, Any]]: 추출된 프레임 정보 목록, 각 항목은 NumPy 배열 이미지 포함
    """
    try:
        # 임시 파일에 비디오 데이터 저장
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(video_buffer)
            temp_path = temp_file.name
        
        try:
            # 프레임 추출
            return extract_frames_from_file(temp_path, interval_sec)
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        logger.error(f"버퍼에서 프레임 추출 중 오류 발생: {e}")
        return []

def preprocess_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    OCR 성능 향상을 위한 이미지 전처리
    
    Args:
        image (np.ndarray): 원본 이미지
    
    Returns:
        np.ndarray: 전처리된 이미지
    """
    try:
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 노이즈 제거
        denoised = cv2.GaussianBlur(binary, (3, 3), 0)
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    except Exception as e:
        logger.warning(f"이미지 전처리 중 오류 발생: {e}, 원본 이미지 반환")
        return image

def crop_image(image: np.ndarray, box: List[float]) -> np.ndarray:
    """
    이미지에서 지정된 박스 영역을 추출합니다.
    
    Args:
        image (np.ndarray): 원본 이미지
        box (List[float]): 박스 좌표 [x1, y1, x2, y2]
    
    Returns:
        np.ndarray: 추출된 이미지 영역
    """
    try:
        x1, y1, x2, y2 = map(int, box)
        
        # 이미지 경계 확인
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # 영역 크기 확인
        if x2 <= x1 or y2 <= y1:
            logger.warning("유효하지 않은 박스 좌표")
            return np.array([])
        
        # 영역 추출
        return image[y1:y2, x1:x2]
    
    except Exception as e:
        logger.error(f"이미지 영역 추출 중 오류 발생: {e}")
        return np.array([])
