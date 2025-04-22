from django.db import models


class StoreReview(models.Model):
    store_name = models.TextField()
    category = models.TextField()
    address = models.TextField()
    visitor_reviews = models.IntegerField()
    blog_reviews = models.IntegerField()
    description_or_menu = models.TextField()

    class Meta:
        managed = False 
        db_table = 'store_reviews'


class StoreReviewText(models.Model):
    store = models.ForeignKey(StoreReview, on_delete=models.CASCADE, db_column='store_id')
    review_text = models.TextField()

    class Meta:
        managed = False  
        db_table = 'store_review_texts'



class Video(models.Model):
    url = models.TextField(unique=True)
    region = models.TextField()
    upload_date = models.DateField()
    processed = models.BooleanField(default=False)

    class Meta:
        db_table = 'videos'


class VideoStoreLink(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    store = models.ForeignKey(StoreReview, on_delete=models.CASCADE)

    class Meta:
        db_table = 'video_store_links'


class VideoStoreSummary(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, db_column='video_id')
    store = models.ForeignKey(StoreReview, on_delete=models.CASCADE, db_column='store_id')
    keyword = models.TextField()
    review_1 = models.TextField()
    review_2 = models.TextField(blank=True, null=True)
    review_3 = models.TextField(blank=True, null=True)

    class Meta:
        managed = False  # 마이그레이션 안할라고 적어둠
        db_table = 'video_store_summaries'
        unique_together = ('video', 'store')

class AnalysisResult(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE, db_column='video_id')
    result_json = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = False 
        db_table = 'analysis_result'
