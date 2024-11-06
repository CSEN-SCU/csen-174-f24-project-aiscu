from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Chat(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username}: {self.message}'

#DevOps
class Counters(models.Model):
    counter = models.IntegerField(default=0)
    index_name = models.TextField(unique=True)

class AvgResponseTime(models.Model):
    time = models.FloatField(default=0)
    total = models.IntegerField(default=0)

class AvgChatLength(models.Model):
    length = models.IntegerField(default=0)
    total = models.IntegerField(default=0)