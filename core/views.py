from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.utils import timezone
from django.http import JsonResponse
import json, requests
from decouple import config
from .models import Video, VideoStoreSummary, StoreReview
from .utils import response, is_valid_url, extract_video_id, is_youtube_video_exists

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

        # ✅ Video 생성 또는 조회
        video, created = Video.objects.get_or_create(
            url=url,
            defaults={
                "upload_date": timezone.now().date(),
                "region": str(region_code),
                "processed": False
            }
        )

        # ✅ 기존 데이터가 있다면 바로 반환
        existing_summaries = VideoStoreSummary.objects.filter(video=video)
        if existing_summaries.exists():
            data = [{
                "keyword": s.keyword,
                "store_id": s.store.id,
                "store_name": s.store.store_name,
                "confidence": None,
                "review_1": s.review_1,
                "review_2": s.review_2,
                "review_3": s.review_3,
            } for s in existing_summaries]
            return response(True, 200, "DB에서 분석 결과 반환", {"data": data})

        # ✅ FastAPI로 분석 요청
        payload = {
            "url": url,
            "video_id": video.id,
            "region_code": region_code
        }

        res = requests.post(MODEL_SERVER_API, json=payload)
        res.raise_for_status()
        result_data = res.json()
        summaries = result_data.get("data", [])

        # ✅ 저장 처리
        unique_store_ids = set()
        saved = []
        for summary in summaries:
            store_id = summary.get("store_id")
            if not store_id or store_id in unique_store_ids:
                continue
            unique_store_ids.add(store_id)

            store, _ = StoreReview.objects.get_or_create(
                id=store_id,
                defaults={"store_name": summary.get("store_name", "이름없음"),
                          "category": "", "address": "", "visitor_reviews": 0,
                          "blog_reviews": 0, "description_or_menu": ""}
            )

            VideoStoreSummary.objects.create(
                video=video,
                store=store,
                keyword=summary.get("keyword"),
                review_1=summary.get("review_1"),
                review_2=summary.get("review_2"),
                review_3=summary.get("review_3"),
            )
            saved.append(summary)

        video.processed = True
        video.save()

        return response(True, 200, "모델 분석 완료 및 DB 저장", {"data": saved})

    except requests.RequestException as e:
        return response(False, 500, f"FastAPI 요청 오류: {str(e)}")
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

        try:
            video = Video.objects.get(id=video_id)
        except Video.DoesNotExist:
            return JsonResponse({"error": "해당 video_id가 존재하지 않습니다."}, status=404)

        # ✅ 중복 제거 + 저장
        unique_store_ids = set()
        saved = []

        # 리스트든 딕셔너리든 처리
        if isinstance(result, dict):
            summaries = result.get("data", [])
        elif isinstance(result, list):
            summaries = result
        else:
            return JsonResponse({"error": "모델 응답 형식 오류"}, status=500)

        for summary in summaries:
            if not isinstance(summary, dict):
                continue

            store_id = summary.get("store_id")
            if not store_id or store_id in unique_store_ids:
                continue
            unique_store_ids.add(store_id)

            store, _ = StoreReview.objects.get_or_create(
                id=store_id,
                defaults={"store_name": summary.get("store_name", "이름없음"),
                          "category": "", "address": "", "visitor_reviews": 0,
                          "blog_reviews": 0, "description_or_menu": ""}
            )

            VideoStoreSummary.objects.create(
                video=video,
                store=store,
                keyword=summary.get("keyword"),
                review_1=summary.get("review_1"),
                review_2=summary.get("review_2"),
                review_3=summary.get("review_3"),
            )
            saved.append(summary)

        # 처리 완료 상태로 업데이트
        video.processed = True
        video.save()

        return JsonResponse({"status": "success", "message": "결과 저장 완료", "data": saved})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
