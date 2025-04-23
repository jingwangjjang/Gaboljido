import os
import re
import cv2
import subprocess
import logging
import numpy as np
import torch
from typing import List, Dict, Any, Optional
import glob
from yt_dlp import YoutubeDL

logger = logging.getLogger("subtitle-detector.utils")


def convert_av1_to_h264(input_path, output_path):
    try:
        subprocess.run([
            "ffmpeg",
            "-i", input_path,
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "fast",
            output_path
        ], check=True)
        print(f"✅ 변환 성공: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg 변환 실패: {e}")
        return None


def download_youtube_video(url: str) -> Optional[Dict[str, Any]]:
    try:
        url = convert_youtube_url(url)
        if not url:
            logger.error("유효하지 않은 YouTube URL입니다.")
            return None

        logger.info(f"YouTube 영상 및 자막 다운로드 중: {url}")

        base_dir = "/home/gynovzs/ocr_videos"
        os.makedirs(base_dir, exist_ok=True)
        output_template = os.path.join(base_dir, "video.%(ext)s")

        #cmd = [
        #    "yt-dlp", url,
        #    "-f", "299+140",  # ✔ AV1 제외
        #    "--write-auto-sub",
        #    "--sub-lang", "ko,en",
        #    "--convert-subs", "srt",
        #    "--output", output_template,
        #    "--quiet"
        #]

        ydl_opts = {
            "format": "bestvideo[ext=mp4][vcodec^=avc1]",
            "outtmpl": output_template,
            "quiet": True,
            "noplaylist": True,
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            print(f"❌ yt-dlp 다운로드 실패: {e}")
            return None

        # 다운로드된 영상 파일 찾기
        video_files = glob.glob(os.path.join(base_dir, "video.mp4"))
        if not video_files:
            print("❌ 영상 다운로드 실패 (파일 없음)")
            return None

        video_path = video_files[0]

        # ▶️ OpenCV 로드
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

        audio_files = glob.glob(os.path.join(base_dir, "video*.mp3"))
        sub_files = glob.glob(os.path.join(base_dir, "video*.srt"))

        audio_final = audio_files[0] if audio_files else None
        sub_final = sub_files[0] if sub_files else None

        return {
            "video_path": video_path,
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
    youtube_patterns = [
        r'(https?://)?(www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(https?://)?(www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
        r'(https?://)?(www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})'
    ]
    
    for pattern in youtube_patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(3)
            return f"https://www.youtube.com/watch?v={video_id}"
    
    return None


def extract_frames_from_video(video_path: str, interval_sec: float = 1.5) -> List[Dict[str, Any]]:
    if not os.path.exists(video_path):
        logger.error(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("비디오 파일을 열 수 없습니다.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval_sec)
    if frame_interval <= 0:
        frame_interval = 1

    logger.info(f"FPS: {fps:.2f}")
    logger.info(f"프레임 추출 간격: {interval_sec:.1f}초 ({frame_interval}프레임마다)")
    logger.info(f"비디오 정보: {total_frames}프레임")

    frames_info = []
    frame_idx = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_info.append({
                "frame_number": frame_idx,
                "timestamp": timestamp,
                "image": frame_rgb
            })

            if len(frames_info) % 10 == 0:
                logger.info(f"프레임 추출 중... 현재 {len(frames_info)}개")

        frame_idx += 1

    cap.release()
    logger.info(f"프레임 추출 완료: 총 {len(frames_info)}개 프레임")
    return frames_info



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
