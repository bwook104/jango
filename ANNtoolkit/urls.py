from django.urls import path

from . import views

urlpatterns = [
    path('main/', views.main, name='main'),

    path('main/result', views.result, name='result'),

]