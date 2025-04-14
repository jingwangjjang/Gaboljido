from django.urls import path
from core.views import analyze_url

urlpatterns = [
    path("analyze-url/", analyze_url),
]
