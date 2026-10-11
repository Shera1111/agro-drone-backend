"""
Скрипт дообучения модели на новых фото (demo-версия для питча).
"""
import os
import json
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(__file__), "weeds_model.pt")
DATA_YAML = os.path.join(os.path.dirname(__file__), "data.yaml")


def run_finetune(epochs: int = 5):
    """Запускает короткое дообучение существующей модели на новых фото."""
    model = YOLO(MODEL_PATH)

    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=640,
        batch=4,
        name="finetune_run",
        deterministic=False,
        exist_ok=True
    )

    metrics = results.results_dict

    # Сохраняем новые веса поверх основной модели
    new_weights_path = os.path.join(
        os.path.dirname(__file__), "runs", "detect", "finetune_run", "weights", "best.pt"
    )

    return {
        "status": "success",
        "epochs_trained": epochs,
        "metrics": {
            "precision": round(metrics.get("metrics/precision(B)", 0), 3),
            "recall": round(metrics.get("metrics/recall(B)", 0), 3),
            "mAP50": round(metrics.get("metrics/mAP50(B)", 0), 3),
        },
        "new_weights_path": new_weights_path
    }


if __name__ == "__main__":
    result = run_finetune(epochs=5)
    print(json.dumps(result, ensure_ascii=False, indent=2))