"""
Agro-Drone AI — основной сервер (Flask)
Содержит два маршрута: анализ фото и дообучение модели.
"""
import json
import os
import io
from flask import Flask, request, Response
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "weeds_model.pt")
DATA_YAML = os.path.join(os.path.dirname(__file__), "data.yaml")

model = YOLO(MODEL_PATH)

CORS_HEADERS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type'
}


def json_response(data, status=200):
    return Response(
        json.dumps(data, ensure_ascii=False),
        status=status,
        headers={**CORS_HEADERS, 'Content-Type': 'application/json; charset=utf-8'}
    )


@app.route('/', methods=['POST', 'OPTIONS'])
def analyze_crop_health():
    if request.method == 'OPTIONS':
        return ('', 204, CORS_HEADERS)

    if 'image' not in request.files:
        return json_response({"error": "Изображение не найдено в запросе (ожидается поле 'image')"}, 400)

    image_file = request.files['image']
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

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
            "box": {"x_min": round(x1, 1), "y_min": round(y1, 1), "x_max": round(x2, 1), "y_max": round(y2, 1)}
        })

    result = {
        "detections": detections,
        "detections_count": len(detections),
        "image_size": {"width": image.width, "height": image.height}
    }

    return json_response(result)


@app.route('/finetune', methods=['POST', 'OPTIONS'])
def finetune():
    if request.method == 'OPTIONS':
        return ('', 204, CORS_HEADERS)

    epochs = int(request.form.get('epochs', 5))

    result = model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=640,
        batch=4,
        name="finetune_run",
        deterministic=False,
        exist_ok=True
    )

    metrics = result.results_dict

    return json_response({
        "status": "success",
        "epochs_trained": epochs,
        "metrics": {
            "precision": round(metrics.get("metrics/precision(B)", 0), 3),
            "recall": round(metrics.get("metrics/recall(B)", 0), 3),
            "mAP50": round(metrics.get("metrics/mAP50(B)", 0), 3),
        }
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)