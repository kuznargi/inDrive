from django.shortcuts import render, redirect
from django.contrib.auth import login,logout
from .models import CustomUser
from .utils import get_face_embedding
import numpy as np
import cv2
from deepface import DeepFace
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
from django.conf import settings
from .gemini_client import analyze_with_gemini
import logging
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)

# Load PyTorch models
def load_pytorch_models():
    """Load dirt and damage detection models"""
    try:
        # Try to load models directly first to see their structure
        dirt_model_path = os.path.join(settings.STATIC_ROOT or settings.BASE_DIR, 'static', 'model', 'dirt_model.pt')
        damage_model_path = os.path.join(settings.STATIC_ROOT or settings.BASE_DIR, 'static', 'model', 'damage_model.pt')
        
        print(f"Loading models from:")
        print(f"Dirt model: {dirt_model_path}")
        print(f"Damage model: {damage_model_path}")
        
        # Check if files exist
        if not os.path.exists(dirt_model_path):
            print(f"Dirt model file not found: {dirt_model_path}")
            return None, None
        if not os.path.exists(damage_model_path):
            print(f"Damage model file not found: {damage_model_path}")
            return None, None
        
        # Try to load the models directly (they might be complete models, not just state_dict)
        try:
            dirt_model = torch.load(dirt_model_path, map_location='cpu')
            damage_model = torch.load(damage_model_path, map_location='cpu')
            
            # Set to evaluation mode
            dirt_model.eval()
            damage_model.eval()
            
            print("Models loaded successfully as complete models")
            return dirt_model, damage_model
            
        except Exception as e:
            print(f"Failed to load as complete models: {e}")
            
            # If that fails, try loading as state_dict with custom architecture
            class CarConditionModel(nn.Module):
                def __init__(self, num_classes=2):
                    super(CarConditionModel, self).__init__()
                    # Simple CNN architecture
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(32, 64, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(64, 128, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool2d((7, 7))
                    )
                    self.classifier = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(128 * 7 * 7, 512),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5),
                        nn.Linear(512, num_classes)
                    )
                
                def forward(self, x):
                    x = self.features(x)
                    x = torch.flatten(x, 1)
                    x = self.classifier(x)
                    return x
            
            dirt_model = CarConditionModel(num_classes=2)
            damage_model = CarConditionModel(num_classes=2)
            
            dirt_model.load_state_dict(torch.load(dirt_model_path, map_location='cpu'))
            damage_model.load_state_dict(torch.load(damage_model_path, map_location='cpu'))
            
            dirt_model.eval()
            damage_model.eval()
            
            print("Models loaded successfully with custom architecture")
            return dirt_model, damage_model
            
    except Exception as e:
        print(f"Error loading PyTorch models: {e}")
        logger.error(f"Error loading PyTorch models: {e}")
        return None, None

# Global model variables
DIRT_MODEL, DAMAGE_MODEL = load_pytorch_models()

def preprocess_image(image_path):
    """Preprocess image for PyTorch model without torchvision"""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize to 224x224
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # Convert to tensor format (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Add batch dimension
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        
        return img_tensor
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        return None

def predict_with_pytorch(image_path):
    """Predict car condition using PyTorch models"""
    try:
        if DIRT_MODEL is None or DAMAGE_MODEL is None:
            logger.error("PyTorch models not loaded")
            return {"error": "Models not loaded"}
        
        # Preprocess image
        input_tensor = preprocess_image(image_path)
        if input_tensor is None:
            logger.error("Image preprocessing failed")
            return {"error": "Image preprocessing failed"}
        
        # Predictions
        with torch.no_grad():
            dirt_output = DIRT_MODEL(input_tensor)
            damage_output = DAMAGE_MODEL(input_tensor)
            
            dirt_pred = torch.softmax(dirt_output, dim=1)
            damage_pred = torch.softmax(damage_output, dim=1)
            
            logger.info(f"Dirt prediction raw: {dirt_pred}")
            logger.info(f"Damage prediction raw: {damage_pred}")
            
            # Get predictions with Russian labels and percentages
            dirt_confidence = float(torch.max(dirt_pred[0])) * 100
            damage_confidence = float(torch.max(damage_pred[0])) * 100
            
            # Determine cleanliness (0=clean, 1=dirty)
            is_dirty = dirt_pred[0][1] > dirt_pred[0][0]
            cleanliness = "Грязная" if is_dirty else "Чистая"
            dirt_percentage = float(dirt_pred[0][1]) * 100 if is_dirty else float(dirt_pred[0][0]) * 100
            
            # Determine integrity (0=intact, 1=damaged)  
            is_damaged = damage_pred[0][1] > damage_pred[0][0]
            integrity = "Поврежден" if is_damaged else "Целый"
            damage_percentage = float(damage_pred[0][1]) * 100 if is_damaged else float(damage_pred[0][0]) * 100
            
            result = {
                "cleanliness": cleanliness,
                "cleanliness_percentage": f"{dirt_percentage:.2f}%",
                "integrity": integrity,
                "integrity_percentage": f"{damage_percentage:.2f}%",
                "cleanliness_raw": "dirty" if is_dirty else "clean",
                "integrity_raw": "damaged" if is_damaged else "intact"
            }
            
            logger.info(f"PyTorch prediction result: {result}")
            return result
            
    except Exception as e:
        logger.error(f"PyTorch prediction error: {e}")
        return {"error": str(e)}

@csrf_exempt
def predict_api(request):
    """API endpoint for car condition prediction"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    try:
        # Check if image file is provided
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image file provided'}, status=400)
        
        uploaded_file = request.FILES['image']
        
        # Save uploaded file temporarily
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
        os.makedirs(upload_dir, exist_ok=True)
        
        file_name = default_storage.save(
            f'temp/{uploaded_file.name}', 
            ContentFile(uploaded_file.read())
        )
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)
        
        # Get PyTorch model predictions
        local_result = predict_with_pytorch(file_path)
        
        # Get Gemini analysis
        gemini_result = None
        try:
            gemini_result = analyze_with_gemini(file_path)
            logger.info(f"Gemini analysis completed for {uploaded_file.name}")
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
        
        # Combine results
        response_data = {
            "local_model": local_result,
            "gemini": gemini_result if gemini_result else None
        }
        
        # Clean up temporary file
        try:
            os.remove(file_path)
        except:
            pass
        
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"Prediction API error: {e}")
        return JsonResponse({'error': str(e)}, status=500)

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
        
        if encoding is None:
            return render(request, 'accounts/signup.html', {'error': 'Face not detected. Please try again.'})

        user = CustomUser.objects.create_user(
            username=username,
            password=password,
            email=email,
            age=age,
            gender=gender
        )
        user.face_encoding = encoding
        user.save()
        login(request, user)
        return redirect('home')

    return render(request, 'accounts/signup.html')

# ---------------- FACE LOGIN ----------------
def face_login(request):
    print(f"face_login called with method: {request.method}")
    print(f"POST data: {request.POST}")
    print(f"FILES data: {list(request.FILES.keys())}")
    
    if request.method == 'POST':
        if 'face_image' not in request.FILES:
            print("No face_image in FILES")
            return render(request, 'accounts/login.html', {'error': 'Пожалуйста, сначала сфотографируйте лицо'})
            
        image_file = request.FILES['face_image']
        print(f"Image file size: {image_file.size}")
        
        try:
            img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            print(f"Image shape: {img.shape if img is not None else 'None'}")

            result = DeepFace.represent(img, model_name="Facenet", enforce_detection=False)
            if not result:
                print("No face detected by DeepFace")
                return render(request, 'accounts/login.html', {'error': 'Лицо не обнаружено'})

            new_encoding = np.array(result[0]["embedding"], dtype=np.float32)
            print(f"Face encoding generated, shape: {new_encoding.shape}")

            # Проверяем совпадение с пользователями
            users_with_faces = CustomUser.objects.exclude(face_encoding__isnull=True)
            print(f"Users with faces count: {users_with_faces.count()}")
            
            if not users_with_faces.exists():
                return render(request, 'accounts/login.html', {'error': 'В системе нет зарегистрированных пользователей с лицами. Пожалуйста, зарегистрируйтесь сначала.'})
            
            best_similarity = 0
            for user in users_with_faces:
                stored_encoding = np.frombuffer(user.face_encoding, dtype=np.float32)
                print(f"Comparing with user: {user.username}")

                # Косинусное сходство
                cosine = np.dot(stored_encoding, new_encoding) / (
                    np.linalg.norm(stored_encoding) * np.linalg.norm(new_encoding)
                )
                print(f"Cosine similarity: {cosine}")
                
                if cosine > best_similarity:
                    best_similarity = cosine

                if cosine > 0.7:  # порог подбирается
                    print(f"Login successful for user: {user.username}")
                    login(request, user)
                    return redirect('home')

            print(f"No matching face found. Best similarity: {best_similarity}")
            if best_similarity < 0.3:
                return render(request, 'accounts/login.html', {'error': 'Лицо не распознано. Убедитесь, что вы зарегистрированы в системе и попробуйте еще раз.'})
            else:
                return render(request, 'accounts/login.html', {'error': f'Пользователь не распознан. Проверьте освещение, посмотрите прямо в камеру и попробуйте еще раз.'})
            
        except Exception as e:
            print(f"Exception in face_login: {e}")
            logger.error(f"Face login error: {e}")
            return render(request, 'accounts/login.html', {'error': 'Ошибка при обработке изображения'})

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

def upload_file(request):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Authentication required'}, status=401)
    
    if request.method == 'POST':
        try:
            uploaded_file = request.FILES.get('file')
            if not uploaded_file:
                return JsonResponse({'error': 'No file provided'}, status=400)
            
            # Create uploads directory if it doesn't exist
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            
            # Save the file
            file_name = default_storage.save(
                f'uploads/{uploaded_file.name}', 
                ContentFile(uploaded_file.read())
            )
            
            # Get full file path for analysis
            file_path = os.path.join(settings.MEDIA_ROOT, file_name)
            
            # Get PyTorch model predictions
            local_result = predict_with_pytorch(file_path)
            logger.info(f"PyTorch analysis completed for {uploaded_file.name}")
            
            # Analyze image with Gemini
            gemini_result = None
            try:
                gemini_result = analyze_with_gemini(file_path)
                logger.info(f"Gemini analysis completed for {uploaded_file.name}")
            except Exception as e:
                logger.error(f"Gemini analysis failed: {e}")
            
            # Combine results for session storage
            combined_results = {
                "local_model": local_result,
                "gemini": gemini_result if gemini_result else None
            }
            
            # Store results in session for results page
            request.session['analysis_results'] = combined_results
            request.session['analyzed_image_url'] = settings.MEDIA_URL + file_name
            request.session['analyzed_image_name'] = uploaded_file.name
            
            response_data = {
                'success': True,
                'message': 'File uploaded and analyzed successfully',
                'file_name': uploaded_file.name,
                'file_size': uploaded_file.size,
                'file_path': file_name,
                'redirect_url': '/results/'
            }
            
            # Add analysis results
            response_data['analysis'] = combined_results
            
            return JsonResponse(response_data)
            
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def results_page(request):
    if not request.user.is_authenticated:
        return redirect('login')
    
    # Get analysis results from session
    analysis_results = request.session.get('analysis_results')
    image_url = request.session.get('analyzed_image_url')
    image_name = request.session.get('analyzed_image_name')
    
    context = {
        'analysis_results': analysis_results,
        'image_url': image_url,
        'image_name': image_name
    }
    
    return render(request, 'results.html', context)