# core/models.py

from django.db import models

class StoreReview(models.Model):
    store_name = models.TextField()
    category = models.TextField()
    address = models.TextField()
    visitor_reviews = models.IntegerField()
    blog_reviews = models.IntegerField()
    description_or_menu = models.TextField()

    def __str__(self):
        return self.store_name
