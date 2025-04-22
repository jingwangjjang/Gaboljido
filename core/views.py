from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.utils import timezone
from django.http import JsonResponse
import json, requests
from decouple import config
from .models import AnalysisResult, Video
from .utils import response, is_valid_url, extract_video_id, is_youtube_video_exists

# .env에 있는 FastAPI 주소 불러오기
MODEL_SERVER_API = config("MODEL_SERVER_API")


@csrf_exempt
@require_POST
def analyze_url(request):
    try:
        body = json.loads(request.body)
        url = body.get("url")
        region_code = body.get("region_code") 

        if not url:
            return response(False, 400, "url은 필수입니다.")
        if region_code is None:
            return response(False, 400, "region_code는 필수입니다.")
        try:
            region_code = int(region_code)
        except ValueError:
            return response(False, 400, "region_code는 정수여야 합니다.")

        if not is_valid_url(url):
            return response(False, 400, "올바르지 않은 유튜브 URL입니다.")

        video_key = extract_video_id(url)
        if not video_key or not is_youtube_video_exists(video_key):
            return response(False, 404, "존재하지 않는 유튜브 Shorts 영상입니다.")

        # ✅ Video 테이블에서 가져오거나 새로 생성
        video, created = Video.objects.get_or_create(
            url=url,
            defaults={
                "upload_date": timezone.now().date(),
                "processed": False
            }
        )

        # ✅ FastAPI로 보낼 요청 바디에 video_id 포함
        payload = {
            "url": url,
            "region_code": region_code,
            "video_id": video.id  # 🔥 중요!
        }

        response_fastapi = requests.post(MODEL_SERVER_API, json=payload)
        response_fastapi.raise_for_status()
        result_data = response_fastapi.json()

        # ✅ 결과 저장
        AnalysisResult.objects.create(video=video, result_json=result_data)

        # ✅ 응답 전송
        return response(True, 200, "모델 분석 완료", result_data)

    except requests.RequestException as e:
        return response(False, 500, f"FastAPI 요청 중 오류 발생: {str(e)}")
    except Exception as e:
        return response(False, 500, f"서버 오류: {str(e)}")


@csrf_exempt
@require_POST
def handle_analysis_result(request):
    try:
        body = json.loads(request.body)
        video_id = body.get("video_id")
        url = body.get("url")
        result = body.get("result")

        if not video_id or not result:
            return JsonResponse({"error": "video_id와 result는 필수입니다."}, status=400)

        video = Video.objects.get(id=video_id)
        AnalysisResult.objects.create(video=video, result_json=result)

        return JsonResponse({"message": "결과 잘 받았음!", "result": result})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
