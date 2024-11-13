from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('chat', views.chatbot, name='chatbot'),
    path('index', views.index, name='index'),
    path('clear', views.clear, name='clear'),
]