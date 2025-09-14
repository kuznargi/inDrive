import logging

logger = logging.getLogger(__name__)

class_names = ['clean', 'dirt-clean-areas']

def process_yolo_results(results):
    detections = []
    try:
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Получаем индекс класса
                cls = int(box.cls[0])
                # Получаем уверенность модели
                conf = float(box.conf[0])
                # Получаем название класса
                class_name = class_names[cls]
                
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': box.xyxy[0].tolist()
                })
        logger.debug(f"Processed detections: {detections}")
    except Exception as e:
        logger.error(f"Error processing YOLO results: {str(e)}")
    return detections