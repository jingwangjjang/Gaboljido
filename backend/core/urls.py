from django.urls import path
from . import views

urlpatterns = [
    path("analyze-url/", views.analyze_url),  # 프론트 → 백 → 모델 서버
    path("result-handler/", views.handle_analysis_result),  # 모델 서버 → 백 (결과 수신용)
]
