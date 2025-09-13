import google.generativeai as genai
import json
import os
from PIL import Image
from django.conf import settings
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your-api-key-here')
genai.configure(api_key=GEMINI_API_KEY)

def analyze_with_gemini(image_path: str) -> dict:
    """
    Analyze car photo using Google Gemini AI
    
    Args:
        image_path (str): Path to the uploaded image file
        
    Returns:
        dict: Analysis results or None if error occurs
    """
    try:
        # Initialize the model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Open and prepare the image
        image = Image.open(image_path)
        
        # Create the prompt for car analysis
        prompt = """
        Ты получаешь фотографию автомобиля. 
        Определи:
        - цвет
        - марку/модель, если можно распознать
        - описание повреждений (коротко, если есть)
        - общую оценку состояния автомобиля (отличное/хорошее/удовлетворительное/плохое)
        - рекомендации по обслуживанию или ремонту
        - дополнительные детали (например, наклейки, багажник открыт, тонировка)
        Верни ответ строго в JSON формате без дополнительного текста.
        
        Пример ответа:
        {
          "color": "белый",
          "brand_model": "Toyota Camry",
          "damage_description": "царапина на переднем бампере",
          "overall_condition": "хорошее",
          "recommendations": "рекомендуется устранить царапину и провести полировку",
          "additional_notes": "тонированные стекла"
        }
        """
        
        # Generate content with the image and prompt
        response = model.generate_content([prompt, image])
        
        # Parse the response
        response_text = response.text.strip()
        
        # Try to extract JSON from the response
        try:
            # Remove any markdown formatting if present
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            # Parse JSON
            analysis_result = json.loads(response_text)
            
            # Validate required fields
            required_fields = ['color', 'brand_model', 'damage_description', 'overall_condition', 'recommendations', 'additional_notes']
            for field in required_fields:
                if field not in analysis_result:
                    analysis_result[field] = ""
            
            logger.info(f"Successfully analyzed image: {image_path}")
            return analysis_result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text}")
            
            # Return a default structure if JSON parsing fails
            return {
                "color": "unknown",
                "brand_model": "unknown",
                "damage_description": "Не удалось проанализировать",
                "overall_condition": "unknown",
                "recommendations": "unknown",
                "additional_notes": f"Ошибка анализа: {response_text[:100]}..."
            }
            
    except Exception as e:
        logger.error(f"Error analyzing image with Gemini: {e}")
        return None

def test_gemini_connection() -> bool:
    """
    Test if Gemini API is properly configured and accessible
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello, this is a test.")
        return True
    except Exception as e:
        logger.error(f"Gemini connection test failed: {e}")
        return False
