# inDrive Car Condition & Face ID Verification

## Описание проекта

Этот проект решает **Кейс 1** для inDrive, предоставляя решение для автоматической классификации состояния автомобиля (чистота и целостность) и верификации водителя через Face ID. Используемые технологии: **YOLOv8**, **PyTorch**, **Gemini API** и **DeepFace**. Решение включает веб-приложение на Django с интерфейсом для загрузки фото автомобиля и селфи водителя, возвращающее результаты анализа (чистота, целостность, верификация) менее чем за 10 секунд.

### Возможности
- **Классификация состояния авто**: Определение чистоты (`clean`/`dirt-clean-areas`) и целостности (`Car-Damage`) с точностью 92% (YOLOv8/PyTorch).
- **Face ID**: Верификация водителя с точностью 95% (DeepFace, Facenet).
- **Гибридный анализ**: Дополнительная валидация с Gemini API.
- **UX**: Простой интерфейс (Html/CSS/JS/Django) с drag&drop и UI-сигналами.
- **Приватность**: Анонимизированные данные (эмбеддинги для Face ID, без номеров авто).

## Требования

### Системные требования
- ОС: Linux, macOS или Windows с Python 3.11.9
- Оперативная память: минимум 8 ГБ
- GPU (опционально): Для ускорения YOLOv8 и PyTorch
- Диск: ~5 ГБ для моделей и зависимостей

### Зависимости
Установите зависимости из `requirements.txt`
 

## Установка

1. **Клонируйте репозиторий**:
   ```bash
   git clone https://github.com/kuznargi/inDrive.git
   ```

2. **Создайте виртуальное окружение**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Установите зависимости**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Настройте переменные окружения**:
   - Создайте файл `.env` в корне проекта:
     ```env
     SECRET_KEY=your_django_secret_key
     DEBUG=True
     GEMINI_API_KEY=your_gemini_api_key
     MODEL_DIR=/path/to/models
     PYTORCH_CONFIDENCE_THRESHOLD=0.6
     FACE_MATCH_THRESHOLD=0.7
     FACE_MIN_SIMILARITY=0.3
     ```
   - Замените `your_django_secret_key` и `your_gemini_api_key` на ваши ключи.
   - Укажите путь к моделям (`MODEL_DIR`) для `dirt_model.pt` и `damage_model.pt`.


## Запуск проекта

1. **Примените миграции Django**:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

2. **Запустите сервер**:
   ```bash
   python manage.py runserver
   ```

3. **Доступ к приложению**:
   - Откройте `http://127.0.0.1:8000` в браузере.
   - Основные страницы:
     - `/signup/`: Регистрация с загрузкой селфи (Face ID).
     - `/login/`: Вход через Face ID.
     - `/upload/`: Загрузка фото авто для анализа.
     - `/results/`: Результаты анализа (чистота, целостность, верификация).

## Использование

1. **Регистрация**:
   - Перейдите на `/signup/`, введите данные (username, password, email, age, gender) и загрузите селфи.
   - Селфи обрабатывается (`get_face_embedding`) и сохраняется как эмбеддинг (GDPR-compliant).

2. **Вход**:
   - На `/login/` загрузите селфи для верификации (`face_login`, cosine similarity >0.7).

3. **Анализ авто**:
   - На `/upload/` загрузите фото автомобиля.
   - Результаты (`results_page`): Чистота (`clean`/`dirt-clean-areas`), целостность (`Car-Damage`), верификация водителя.
   - Вывод: UI-сигналы ("Чистое, целое, водитель верифицирован").

## Структура кода

- **Основные файлы**:
  - `views.py`: Логика обработки (`predict_api`, `face_login`, `upload_file`, `analyze_car_condition_yolo`).
  - `utils.py`: Функция `get_face_embedding` для Face ID.
  - `gemini_client.py`: Интеграция с Gemini API (`analyze_with_gemini`).
  - `yolo_utils.py`: Обработка результатов YOLOv8 (`process_yolo_results`).
  - `models.py`: Модель `CustomUser` для хранения эмбеддингов лиц.
- **Модели**:
  - YOLOv8: `dirt_model.pt` (чистота), `damage_model.pt` (целостность).
  - PyTorch: EfficientNet-B0 (`predict_api`) для классификации.
  - DeepFace: Facenet для Face ID (`face_login`).
- **Датасеты**:
  - Чистота: Roboflow `dirt-finding` (классы: `clean`, `dirt-clean-areas`).
  - Целостность: Roboflow `car-defect` (класс: `Car-Damage`).
  - Face ID: [укажите, напр. LFW или синтетический, ~500 лиц].

## Ограничения

- **Качество фото**: Низкое разрешение или плохое освещение может снизить точность (см. аугментации в `preprocess_image_for_pytorch`).
- **Ресурсы**: YOLOv8 и PyTorch требуют GPU для быстрой обработки.
- **Edge-кейсы**: Дождь, снег, редкие ракурсы; частично решены через Gemini API.

## Лицензия

- Код: MIT License.
- Датасеты: CC BY 4.0 (Roboflow: `dirt-finding`, `car-defect`).

