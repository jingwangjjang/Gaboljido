from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import requests
from django.utils import timezone
from .models import Video
from .utils import is_valid_url, extract_video_id, is_youtube_video_exists, response
from decouple import config

MODEL_SERVER_API = config("MODEL_SERVER_API")


@csrf_exempt
def analyze_url(request):
    if request.method == "POST":
        try:
            body = json.loads(request.body)
            url = body.get("url")
            region = body.get("region")

            if not url:
                return response(False, 400, "url 필드는 필수입니다.")
            if not region:
                return response(False, 400, "region 필드는 필수입니다.")

            if not is_valid_url(url):
                return response(False, 400, "올바르지 않은 URL입니다.")

            video_key = extract_video_id(url)
            if not video_key:
                return response(False, 400, "유튜브 Shorts URL만 지원됩니다.")

            if not is_youtube_video_exists(video_key):
                return response(False, 404, "존재하지 않는 유튜브 Shorts 영상입니다.")

            # DB에 영상 정보 저장 또는 가져오기
            video, created = Video.objects.get_or_create(
                url=url,
                defaults={
                    "region": region,
                    "upload_date": timezone.now().date(),
                    "processed": False
                }
            )

            # 모델 서버로 전달
            payload = {
                "video_id": video.id,
                "url": url,
                "region": region
            }

            requests.post(MODEL_SERVER_API, json=payload)

            return response(True, 200, "모델 분석 요청 완료", payload)

        except Exception as e:
            return response(False, 500, f"서버 오류: {str(e)}")

    return response(False, 405, "허용되지 않은 메서드입니다.")
