import requests
from urllib.parse import urlparse
from django.http import JsonResponse
from decouple import config

# 공통 응답 함수
def response(success=True, code=200, message="", data=None):
    return JsonResponse({
        "status": "success" if success else "error",
        "code": code,
        "message": message,
        "data": data
    }, status=code)

# 유효한 URL인지 확인
def is_valid_url(url: str) -> bool:
    try:
        res = requests.head(url, allow_redirects=True, timeout=5)
        return res.status_code < 400
    except:
        return False

# Shorts 영상 ID 추출 (Shorts만 허용)

def extract_video_id(url: str) -> str:
    parsed_url = urlparse(url)
    if "youtube.com" in parsed_url.netloc and parsed_url.path.startswith("/shorts/"):
        path_parts = parsed_url.path.split("/")
        if len(path_parts) >= 3:
            return path_parts[2]
        elif len(path_parts) == 2:
            return path_parts[-1]
    return None


# 유튜브 영상 존재 여부 확인
YOUTUBE_API_KEY = config('YOUTUBE_API_KEY')

def is_youtube_video_exists(video_id: str) -> bool:
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet",
        "id": video_id,
        "key": YOUTUBE_API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    return len(data.get("items", [])) > 0
