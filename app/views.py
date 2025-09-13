from django.shortcuts import render, redirect
from django.contrib.auth import login,logout
from .models import CustomUser
from .utils import get_face_embedding
import numpy as np
import cv2
from deepface import DeepFace

# ---------------- SIGNUP ----------------
def signup(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        email = request.POST['email']
        age = request.POST['age']
        gender = request.POST['gender']
        image_file = request.FILES['face_image']

        encoding = get_face_embedding(image_file)

        user = CustomUser.objects.create_user(
            username=username,
            password=password,
            email=email,
            age=age,
            gender=gender
        )
        user.save()
        login(request, user)
        return redirect('home')

    return render(request, 'accounts/signup.html')

# ---------------- FACE LOGIN ----------------
def face_login(request):
    if request.method == 'POST':
        image_file = request.FILES['face_image']
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        result = DeepFace.represent(img, model_name="Facenet", enforce_detection=False)
        if not result:
            return render(request, 'accounts/login.html', {'error': 'No face detected'})

        new_encoding = np.array(result[0]["embedding"], dtype=np.float32)

        # Проверяем совпадение с пользователями
        for user in CustomUser.objects.all():
            stored_encoding = np.frombuffer(user.face_encoding, dtype=np.float32)

            # Косинусное сходство
            cosine = np.dot(stored_encoding, new_encoding) / (
                np.linalg.norm(stored_encoding) * np.linalg.norm(new_encoding)
            )

            if cosine > 0.7:  # порог подбирается
                login(request, user)
                return redirect('home')

        return render(request, 'accounts/login.html', {'error': 'Face not recognized'})

    return render(request, 'accounts/login.html')

def user_logout(request):
    logout(request)
    return redirect('login')  

def profile(request):
    if not request.user.is_authenticated:
        return redirect('login')
    return render(request, 'profile.html', {'user': request.user})

def home(request):
    if not request.user.is_authenticated:
        return redirect('login')
    return render(request, 'home.html', {'user': request.user})