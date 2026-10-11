import functions_framework
import json
import os
from flask import jsonify
from ultralytics import YOLO
from PIL import Image
import io

# Загружаем модель один раз при старте (не при каждом запросе — так быстрее)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "weeds_model.pt")
model = YOLO(MODEL_PATH)

@functions_framework.http
def analyze_crop_health_http(request):
    # Разрешаем запросы с любого источника (нужно для фронтенда на Vercel)
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
    }

    if request.method == 'OPTIONS':
        return ('', 204, headers)

    # Проверяем, что в запросе действительно есть файл изображения
    if 'image' not in request.files:
        return (json.dumps({"error": "Изображение не найдено в запросе (ожидается поле 'image')"}, ensure_ascii=False), 400, {**headers, 'Content-Type': 'application/json; charset=utf-8'})

    image_file = request.files['image']
    image_bytes = image_file.read()

    # Открываем изображение через Pillow
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Реальный инференс через YOLO
    results = model(image)

    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

        detections.append({
            "class_name": class_name,
            "confidence": round(confidence, 3),
            "box": {
                "x_min": round(x1, 1),
                "y_min": round(y1, 1),
                "x_max": round(x2, 1),
                "y_max": round(y2, 1)
            }
        })

    result = {
        "detections": detections,
        "detections_count": len(detections),
        "image_size": {"width": image.width, "height": image.height}
    }

    # Сохраняем результат локально для отладки (только не в облаке)
    if not os.environ.get("K_SERVICE"):
        with open("output_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return (json.dumps(result, ensure_ascii=False), 200, {**headers, 'Content-Type': 'application/json; charset=utf-8'})