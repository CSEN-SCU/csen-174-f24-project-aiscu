from django.db import models
#from django.contrib.auth.models import User

# Create your models here.
#class Chat(models.Model):
#    user = models.ForeignKey(User, on_delete=models.CASCADE)
#    message = models.TextField()
#    response = models.TextField()
#    created_at = models.DateTimeField(auto_now_add=True)
#
#    def __str__(self):
#        return f'{self.user.username}: {self.message}'

#DevOps
class DevOpsMetrics(models.Model):
    chatbot_index = models.TextField()
    metric_type = models.TextField()
    metric_value = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

class Counters(models.Model):
    counter = models.IntegerField(default=0)
    index_name = models.TextField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class AvgResponseTime(models.Model):
    time = models.FloatField(default=0)
    total = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class AvgChatLength(models.Model):
    length = models.IntegerField(default=0)
    total = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)