from django.urls import path
from . import views
from django.shortcuts import render

urlpatterns = [
    path('signup/', views.signup, name='signup'),
    path('login/', views.face_login, name='login'),
    path('home/', views.home, name='home'),
    path('logout/', views.user_logout, name='logout'),
    path('', lambda request: render(request, "main.html"), name='main'),
    path('profile/', views.profile, name='profile'),
    path('upload/', views.upload_page, name='upload_page'),
    path('upload-file/', views.upload_file, name='upload_file'),
    path('results/', views.results_page, name='results_page'),
    path('predict/', views.predict_api, name='predict_api'),
]