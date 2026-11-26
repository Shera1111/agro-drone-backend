"""
Agro-Drone AI: сервер (Flask), версия с нарезкой больших снимков на плитки.

Маршруты:
  POST /          анализ фото (поле "image"; необязательно: conf, tiling)
  POST /finetune  короткое дообучение (на бесплатном Render не работает: не хватает памяти)

Модель берётся из переменной окружения MODEL_PATH (по умолчанию weeds_model.pt).
Для проверки у себя на компьютере (PowerShell):
    $env:MODEL_PATH="drone_v1.pt"; python app.py
"""
import io
import json
import os

from flask import Flask, Response, request
from PIL import Image
from ultralytics import YOLO

from batch_analyze import analyze_image, tile_origins

Image.MAX_IMAGE_PIXELS = None

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, os.environ.get("MODEL_PATH", "weeds_model.pt"))
DATA_YAML = os.path.join(BASE_DIR, "data.yaml")

TILE = 640
OVERLAP = 0.2
AUTO_TILE_ABOVE = 1500  # снимки с длинной стороной больше этого режем на плитки
MAX_TILES = int(os.environ.get("MAX_TILES", "150"))  # защита от слишком больших снимков

model = YOLO(MODEL_PATH)

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


def json_response(data, status=200):
    return Response(
        json.dumps(data, ensure_ascii=False),
        status=status,
        headers={**CORS_HEADERS, "Content-Type": "application/json; charset=utf-8"},
    )


def count_tiles(width, height):
    step = max(1, int(TILE * (1 - OVERLAP)))
    return len(tile_origins(width, TILE, step)) * len(tile_origins(height, TILE, step))


def to_detection(box, score):
    x1, y1, x2, y2 = box
    return {
        "class_name": "weeds",
        "confidence": round(float(score), 3),
        "box": {
            "x_min": round(x1, 1),
            "y_min": round(y1, 1),
            "x_max": round(x2, 1),
            "y_max": round(y2, 1),
        },
    }


@app.route("/", methods=["POST", "OPTIONS"])
def analyze_crop_health():
    if request.method == "OPTIONS":
        return ("", 204, CORS_HEADERS)

    if "image" not in request.files:
        return json_response({"error": "Изображение не найдено в запросе (ожидается поле 'image')"}, 400)

    try:
        conf = float(request.form.get("conf", 0.25))
    except ValueError:
        return json_response({"error": "conf должен быть числом, например 0.25"}, 400)
    tiling = request.form.get("tiling", "auto")  # auto | on | off

    data = request.files["image"].read()
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
    except Exception:
        return json_response({"error": "Не удалось открыть файл как изображение"}, 400)

    use_tiles = tiling == "on" or (tiling == "auto" and max(width, height) > AUTO_TILE_ABOVE)

    if use_tiles:
        n_tiles = count_tiles(width, height)
        if n_tiles > MAX_TILES:
            return json_response(
                {"error": f"Снимок слишком большой для этого сервера: {n_tiles} плиток, максимум {MAX_TILES}."},
                413,
            )
        _, found, n_tiles = analyze_image(model, io.BytesIO(data), TILE, OVERLAP, conf, batch=4)
        detections = [to_detection(b, s) for b, s in found]
        mode = "tiles"
    else:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        result = model(image, conf=conf, verbose=False)[0]
        detections = [
            to_detection(b.xyxy[0].tolist(), b.conf[0]) for b in result.boxes
        ]
        n_tiles = 1
        mode = "full"

    return json_response(
        {
            "detections": detections,
            "detections_count": len(detections),
            "image_size": {"width": width, "height": height},
            "tiles": n_tiles,
            "mode": mode,
        }
    )


@app.route("/finetune", methods=["POST", "OPTIONS"])
def finetune():
    if request.method == "OPTIONS":
        return ("", 204, CORS_HEADERS)

    epochs = int(request.form.get("epochs", 5))
    result = model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=640,
        batch=4,
        name="finetune_run",
        deterministic=False,
        exist_ok=True,
    )
    metrics = result.results_dict
    return json_response(
        {
            "status": "success",
            "epochs_trained": epochs,
            "metrics": {
                "precision": round(metrics.get("metrics/precision(B)", 0), 3),
                "recall": round(metrics.get("metrics/recall(B)", 0), 3),
                "mAP50": round(metrics.get("metrics/mAP50(B)", 0), 3),
            },
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
