import os
import re
import cv2
import subprocess
import logging
import tempfile
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger("subtitle-detector.utils")

def download_youtube_video(url: str) -> Optional[Dict[str, Any]]:
    """
    yt-dlp를 사용하여 YouTube 영상을 다운로드하고,
    영상 프레임(np.ndarray), 오디오(mp3), 자막(SRT)을 반환합니다.

    Args:
        url (str): YouTube URL

    Returns:
        Optional[Dict[str, Any]]: {
            'frames': np.ndarray,
            'audio_path': str,
            'subtitle_path': str
        }
    """
    try:
        url = convert_youtube_url(url)
        if not url:
            logger.error("유효하지 않은 YouTube URL입니다.")
            return None

        logger.info(f"YouTube 영상 및 자막 다운로드 중: {url}")

        # 임시 파일 생성 경로
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = os.path.join(tmp_dir, "video.mp4")
            audio_path = os.path.join(tmp_dir, "audio.mp3")
            subtitle_path = os.path.join(tmp_dir, "subtitle.srt")

            # yt-dlp 명령 구성 (영상 + 자막 + 오디오)
            cmd = [
                "yt-dlp",
                url,
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
                "--write-auto-sub",  # 자동 자막 다운로드
                "--sub-lang", "ko,en",  # 한글/영어 우선
                "--convert-subs", "srt",  # 자막을 .srt로 변환
                "--output", os.path.join(tmp_dir, "video.%(ext)s"),
                "--quiet"
            ]

            subprocess.run(cmd, check=True)

            # ▶️ 영상 → NumPy 배열로 로드
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error("다운로드한 비디오 파일을 열 수 없습니다.")
                return None

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            logger.info(f"✅ 총 {len(frames)} 프레임 읽음")

            # 오디오 및 자막 파일 경로 확인
            # 확장자 자동 결정된 파일 찾기
            audio_files = [f for f in os.listdir(tmp_dir) if f.endswith(".mp3")]
            sub_files = [f for f in os.listdir(tmp_dir) if f.endswith(".srt")]

            audio_final = os.path.join(tmp_dir, audio_files[0]) if audio_files else None
            sub_final = os.path.join(tmp_dir, sub_files[0]) if sub_files else None

            return {
                "frames": np.array(frames),
                "audio_path": audio_final,
                "subtitle_path": sub_final
            }

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
