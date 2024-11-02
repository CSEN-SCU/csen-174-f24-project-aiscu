from django.urls import path
from . import views

urlpatterns = [
    path('', views.chatbot, name='chatbot'),
    path('login', views.login, name='login'),
    path('register', views.register, name='register'),
    path('logout', views.logout, name='logout'),
    path('home', views.home, name='home'),
    path('index', views.index, name='index'),
    path('clear', views.clear, name='clear'),
]