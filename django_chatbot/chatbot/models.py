from django.db import models
#from django.contrib.auth.models import User

#DevOps
class DevOpsMetrics(models.Model):
    chatbot_index = models.TextField()
    metric_type = models.TextField()
    metric_value = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)