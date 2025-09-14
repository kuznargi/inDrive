from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings

from .models import CustomUser
from .utils import get_face_embedding
from .gemini_client import analyze_with_gemini, ensure_gemini_api_key_configured

import os
import uuid
import logging
import numpy as np
import cv2
from deepface import DeepFace
from PIL import Image

import torch
import torch.nn as nn
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from mimetypes import guess_extension
from .yolo_utils import process_yolo_results, class_names

logger = logging.getLogger(__name__)


# Module-level caches for lazy loading
_YOLO_MODELS = {'dirt': None, 'damage': None}
_PYTORCH_MODELS = {'dirt': None, 'damage': None}


def get_model_paths():
    base = getattr(settings, 'MODEL_DIR', None)
    if base is None:
        base = os.path.join(settings.BASE_DIR, 'static', 'model')
    base = str(base)
    dirt_model_path = os.path.join(base, 'dirt_model.pt')
    damage_model_path = os.path.join(base, 'damage_model.pt')
    return dirt_model_path, damage_model_path


def load_yolo_models_lazy():
    if YOLO is None:
        return None, None
    if _YOLO_MODELS['dirt'] is None or _YOLO_MODELS['damage'] is None:
        try:
            dirt_path, damage_path = get_model_paths()
            if not os.path.exists(dirt_path) or not os.path.exists(damage_path):
                logger.warning('YOLO model files not found at %s and %s', dirt_path, damage_path)
                return None, None
            _YOLO_MODELS['dirt'] = YOLO(dirt_path)
            _YOLO_MODELS['damage'] = YOLO(damage_path)
            logger.info('YOLO models loaded')
        except Exception as e:
            logger.exception('Error loading YOLO models: %s', e)
            return None, None
    return _YOLO_MODELS['dirt'], _YOLO_MODELS['damage']


def load_pytorch_models_lazy():
    if _PYTORCH_MODELS['dirt'] is None or _PYTORCH_MODELS['damage'] is None:
        try:
            dirt_path, damage_path = get_model_paths()
            if not os.path.exists(dirt_path) or not os.path.exists(damage_path):
                logger.warning('PyTorch model files not found at %s and %s', dirt_path, damage_path)
                return None, None

            try:
                _PYTORCH_MODELS['dirt'] = torch.load(dirt_path, map_location='cpu')
                _PYTORCH_MODELS['damage'] = torch.load(damage_path, map_location='cpu')
            except Exception:
                # If direct load fails, try loading state dict into a small architecture
                class CarConditionModel(nn.Module):
                    def __init__(self, num_classes=2):
                        super().__init__()
                        self.features = nn.Sequential(
                            nn.Conv2d(3, 16, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.AdaptiveAvgPool2d((7, 7)),
                        )
                        self.classifier = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(16 * 7 * 7, 64),
                            nn.ReLU(),
                            nn.Linear(64, num_classes),
                        )

                    def forward(self, x):
                        x = self.features(x)
                        x = self.classifier(x)
                        return x

                d = CarConditionModel()
                g = CarConditionModel()
                d.load_state_dict(torch.load(dirt_path, map_location='cpu'))
                g.load_state_dict(torch.load(damage_path, map_location='cpu'))
                d.eval(); g.eval()
                _PYTORCH_MODELS['dirt'] = d
                _PYTORCH_MODELS['damage'] = g

            _PYTORCH_MODELS['dirt'].eval()
            _PYTORCH_MODELS['damage'].eval()
            logger.info('PyTorch models loaded')
        except Exception as e:
            logger.exception('Error loading PyTorch models: %s', e)
            return None, None
    return _PYTORCH_MODELS['dirt'], _PYTORCH_MODELS['damage']


def is_image_file(uploaded_file) -> bool:
    ctype = getattr(uploaded_file, 'content_type', None)
    if ctype and ctype.startswith('image/'):
        return True
    try:
        Image.open(uploaded_file).verify()
        uploaded_file.seek(0)
        return True
    except Exception:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        return False


def save_uploaded_file_safe(uploaded_file, subdir='uploads'):
    media_root = settings.MEDIA_ROOT
    target_dir = os.path.join(media_root, subdir)
    os.makedirs(target_dir, exist_ok=True)
    ext = os.path.splitext(uploaded_file.name)[1]
    if not ext:
        ext = guess_extension(getattr(uploaded_file, 'content_type', '') or '') or ''
    filename = f"{uuid.uuid4().hex}{ext}"
    rel_path = os.path.join(subdir, filename)
    full_path = os.path.join(media_root, rel_path)
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    default_storage.save(rel_path, ContentFile(uploaded_file.read()))
    return rel_path, full_path, filename


def preprocess_image_for_pytorch(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        arr = np.array(image).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor
    except Exception as e:
        logger.exception('preprocess_image_for_pytorch error: %s', e)
        return None


def analyze_car_condition_yolo(image_path):
    dirt_model, damage_model = load_yolo_models_lazy()
    if dirt_model is None or damage_model is None:
        return {'error': 'YOLO models not available'}

    img = cv2.imread(image_path)
    if img is None:
        return {'error': 'Failed to read image'}

    total_area = img.shape[0] * img.shape[1]
    dirt_area = 0
    damage_area = 0
    is_dirty = False
    is_damaged = False

    try:
        dirt_results = dirt_model(img, conf=0.5, verbose=False)
        dirt_detections = process_yolo_results(dirt_results)
        for det in dirt_detections:
            if det['class'] == 'dirt-clean-areas':
                is_dirty = True
                x1, y1, x2, y2 = map(int, det['bbox'])
                dirt_area += (x2 - x1) * (y2 - y1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        damage_results = damage_model(img, conf=0.5, verbose=False)
        damage_detections = process_yolo_results(damage_results)
        for det in damage_detections:
            if det['class'] == 'dirt-clean-areas':
                is_damaged = True
                x1, y1, x2, y2 = map(int, det['bbox'])
                damage_area += (x2 - x1) * (y2 - y1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        dirt_percent = (dirt_area / total_area * 100) if total_area > 0 else 0
        damage_percent = (damage_area / total_area * 100) if total_area > 0 else 0

        cleanliness = 'Грязная' if is_dirty else 'Чистая'
        integrity = 'Поврежден' if is_damaged else 'Целый'

        result_path = os.path.join(settings.MEDIA_ROOT, 'result.jpg')
        cv2.imwrite(result_path, img)

        # build detection summary counts
        detected = {}
        for name in class_names:
            detected[name] = 0
        for d in dirt_detections + damage_detections:
            detected[d['class']] = detected.get(d['class'], 0) + 1

        return {
            'cleanliness': cleanliness,
            'dirt_percent': float(f"{dirt_percent:.2f}"),
            'integrity': integrity,
            'damage_percent': float(f"{damage_percent:.2f}"),
            'result_image_url': settings.MEDIA_URL + 'result.jpg',
            'detected_classes': detected
        }
    except Exception as e:
        logger.exception('analyze_car_condition_yolo error: %s', e)
        return {'error': str(e)}


@csrf_exempt
def predict_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)

    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image file provided'}, status=400)

    uploaded_file = request.FILES['image']
    if not is_image_file(uploaded_file):
        return JsonResponse({'error': 'Uploaded file is not a valid image'}, status=400)

    try:
        rel_path, file_path, filename = save_uploaded_file_safe(uploaded_file, subdir='temp')

        # Prefer YOLO, fallback to PyTorch
        yolo_dirt, yolo_damage = load_yolo_models_lazy()
        if yolo_dirt and yolo_damage:
            local_result = analyze_car_condition_yolo(file_path)
        else:
            torch_dirt, torch_damage = load_pytorch_models_lazy()
            if torch_dirt and torch_damage:
                tensor = preprocess_image_for_pytorch(file_path)
                if tensor is None:
                    local_result = {'error': 'Image preprocessing failed'}
                else:
                    try:
                        # run both models for consistency with YOLO outputs
                        out_dirt = torch_dirt(tensor)
                        out_damage = torch_damage(tensor)
                        pred_dirt = torch.softmax(out_dirt, dim=1)
                        pred_damage = torch.softmax(out_damage, dim=1)

                        dirt_conf = float(torch.max(pred_dirt[0]).item())
                        damage_conf = float(torch.max(pred_damage[0]).item())
                        # scale to 0-100 for display
                        dirt_conf_pct = dirt_conf * 100 if dirt_conf <= 1.0 else dirt_conf
                        damage_conf_pct = damage_conf * 100 if damage_conf <= 1.0 else damage_conf

                        cleanliness = 'Грязная' if pred_dirt[0][1] > pred_dirt[0][0] else 'Чистая'
                        integrity = 'Поврежден' if pred_damage[0][1] > pred_damage[0][0] else 'Целый'

                        threshold = getattr(settings, 'PYTORCH_CONFIDENCE_THRESHOLD', 0.6)
                        is_confident = (max(dirt_conf, damage_conf) >= threshold)

                        local_result = {
                            'cleanliness': cleanliness,
                            'dirt_percent': f"{dirt_conf_pct:.2f}%",
                            'integrity': integrity,
                            'integrity_confidence': damage_conf_pct,
                            'is_confident': is_confident,
                            'detected_classes': {name: 0 for name in class_names}
                        }
                    except Exception as e:
                        logger.exception('PyTorch prediction error: %s', e)
                        local_result = {'error': str(e)}
            else:
                local_result = {'error': 'No local models available'}

        gemini_result = None
        try:
            if ensure_gemini_api_key_configured():
                gemini_result = analyze_with_gemini(file_path)
        except Exception as e:
            logger.exception('Gemini analysis failed: %s', e)

        response_data = {'local_model': local_result, 'gemini': gemini_result}

        try:
            os.remove(file_path)
        except Exception:
            pass

        return JsonResponse(response_data)
    except Exception as e:
        logger.exception('predict_api error: %s', e)
        return JsonResponse({'error': str(e)}, status=500)


def ensure_authenticated_redirect(request):
    if not request.user.is_authenticated:
        return redirect('login')


def signup(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        email = request.POST.get('email')
        age = request.POST.get('age')
        gender = request.POST.get('gender')
        image_file = request.FILES.get('face_image')

        if not image_file:
            return render(request, 'accounts/signup.html', {'error': 'No face image provided'})

        encoding = get_face_embedding(image_file)
        if encoding is None:
            return render(request, 'accounts/signup.html', {'error': 'Face not detected. Please try again.'})

        user = CustomUser.objects.create_user(username=username, password=password, email=email, age=age, gender=gender)
        user.face_encoding = encoding
        user.save()
        login(request, user)
        return redirect('home')

    return render(request, 'accounts/signup.html')


def face_login(request):
    if request.method == 'POST':
        if 'face_image' not in request.FILES:
            return render(request, 'accounts/login.html', {'error': 'Please provide face image'})

        image_file = request.FILES['face_image']
        try:
            img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            result = DeepFace.represent(img, model_name='Facenet', enforce_detection=False)
            if not result:
                return render(request, 'accounts/login.html', {'error': 'Face not detected'})

            new_encoding = np.array(result[0]['embedding'], dtype=np.float32)

            users_with_faces = CustomUser.objects.exclude(face_encoding__isnull=True)
            if not users_with_faces.exists():
                return render(request, 'accounts/login.html', {'error': 'No users with faces registered'})

            best_similarity = 0.0
            match_threshold = getattr(settings, 'FACE_MATCH_THRESHOLD', 0.7)
            min_similarity = getattr(settings, 'FACE_MIN_SIMILARITY', 0.3)

            for user in users_with_faces:
                stored_encoding = np.frombuffer(user.face_encoding, dtype=np.float32)
                cosine = np.dot(stored_encoding, new_encoding) / (np.linalg.norm(stored_encoding) * np.linalg.norm(new_encoding))
                if cosine > best_similarity:
                    best_similarity = float(cosine)
                if cosine > match_threshold:
                    login(request, user)
                    return redirect('home')

            if best_similarity < min_similarity:
                return render(request, 'accounts/login.html', {'error': 'Face not recognized'})
            return render(request, 'accounts/login.html', {'error': 'User not recognized'})
        except Exception as e:
            logger.exception('face_login error: %s', e)
            return render(request, 'accounts/login.html', {'error': 'Error processing image'})

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


def upload_page(request):
    if not request.user.is_authenticated:
        return redirect('login')
    return render(request, 'upload.html')


@csrf_exempt
def upload_file(request):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Authentication required'}, status=401)
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)

    uploaded_file = request.FILES.get('file')
    if not uploaded_file:
        return JsonResponse({'error': 'No file provided'}, status=400)
    if not is_image_file(uploaded_file):
        return JsonResponse({'error': 'Uploaded file is not a valid image'}, status=400)

    try:
        rel_path, file_path, filename = save_uploaded_file_safe(uploaded_file, subdir='uploads')

        # analyze
        yolo_dirt, yolo_damage = load_yolo_models_lazy()
        if yolo_dirt and yolo_damage:
            local_result = analyze_car_condition_yolo(file_path)
        else:
            torch_dirt, torch_damage = load_pytorch_models_lazy()
            if torch_dirt and torch_damage:
                tensor = preprocess_image_for_pytorch(file_path)
                if tensor is None:
                    local_result = {'error': 'Image preprocessing failed'}
                else:
                    try:
                        out = torch_damage(tensor)
                        pred = torch.softmax(out, dim=1)
                        integrity = 'Поврежден' if pred[0][1] > pred[0][0] else 'Целый'
                        local_result = {'integrity': integrity, 'integrity_confidence': float(torch.max(pred[0])) * 100}
                    except Exception as e:
                        logger.exception('PyTorch analysis error: %s', e)
                        local_result = {'error': str(e)}
            else:
                local_result = {'error': 'No local models available'}

        gemini_result = None
        try:
            if ensure_gemini_api_key_configured():
                gemini_result = analyze_with_gemini(file_path)
        except Exception as e:
            logger.exception('Gemini analysis failed: %s', e)

        combined_results = {'local_model': local_result, 'gemini': gemini_result}

        request.session['analysis_results'] = combined_results
        request.session['analyzed_image_url'] = settings.MEDIA_URL + rel_path
        request.session['analyzed_image_name'] = filename

        response_data = {
            'success': True,
            'message': 'File uploaded and analyzed successfully',
            'file_name': filename,
            'file_size': uploaded_file.size,
            'file_path': rel_path,
            'redirect_url': '/results/',
            'analysis': combined_results
        }
        return JsonResponse(response_data)
    except Exception as e:
        logger.exception('upload_file error: %s', e)
        return JsonResponse({'error': str(e)}, status=500)


def results_page(request):
    if not request.user.is_authenticated:
        return redirect('login')
    analysis_results = request.session.get('analysis_results')
    image_url = request.session.get('analyzed_image_url')
    image_name = request.session.get('analyzed_image_name')
    context = {'analysis_results': analysis_results, 'image_url': image_url, 'image_name': image_name}
    return render(request, 'results.html', context)


def get_condition_state(score):
    if score < 0.3:
        return "Отличное"
    elif score < 0.6:
        return "Среднее"
    else:
        return "Плохое"