"""
Agro-Drone AI: пакетный анализ снимков с дрона.

Каждый большой снимок режется на плитки (tiling), чтобы мелкие сорняки
не терялись при сжатии кадра до 640 пикселей. Рамки с разных плиток
собираются обратно в координаты исходного снимка, дубли убираются (NMS).

Запуск (из папки проекта, с активным venv):
    python batch_analyze.py drone_test
    python batch_analyze.py drone_test --model weeds_model.pt --conf 0.25
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms
from ultralytics import YOLO

try:  # поддержка AVIF, если плагин установлен (pip install pillow-avif-plugin)
    import pillow_avif  # noqa: F401
except ImportError:
    pass

IMG_EXT = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff", ".avif"}
Image.MAX_IMAGE_PIXELS = None  # снимки с дронов бывают очень большими


def tile_origins(length, tile, step):
    """Позиции начала плиток вдоль одной оси. Последняя плитка прижата к краю."""
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, step))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def suppress_contained(detections, thr=0.8):
    """Убирает рамки, которые почти целиком лежат внутри рамки с большей уверенностью.

    Обычный NMS не склеивает такие пары (у маленькой рамки внутри большой
    пересечение с объединением мало), а на стыках плиток они возникают часто.
    """
    kept = []
    for box, score in sorted(detections, key=lambda d: d[1], reverse=True):
        x1, y1, x2, y2 = box
        area = max(1e-9, (x2 - x1) * (y2 - y1))
        duplicate = False
        for (kx1, ky1, kx2, ky2), _ in kept:
            iw = min(x2, kx2) - max(x1, kx1)
            ih = min(y2, ky2) - max(y1, ky1)
            if iw <= 0 or ih <= 0:
                continue
            k_area = max(1e-9, (kx2 - kx1) * (ky2 - ky1))
            if iw * ih / min(area, k_area) >= thr:
                duplicate = True
                break
        if not duplicate:
            kept.append((box, score))
    return kept


def analyze_image(model, path, tile, overlap, conf, batch):
    img = Image.open(path).convert("RGB")
    width, height = img.size
    step = max(1, int(tile * (1 - overlap)))
    origins = [
        (x, y)
        for y in tile_origins(height, tile, step)
        for x in tile_origins(width, tile, step)
    ]

    boxes, scores = [], []
    for i in range(0, len(origins), batch):
        chunk = origins[i : i + batch]
        crops = []
        for x, y in chunk:
            crop = np.asarray(img.crop((x, y, min(x + tile, width), min(y + tile, height))))
            # ultralytics ждёт для numpy-массивов порядок цветов BGR
            crops.append(np.ascontiguousarray(crop[:, :, ::-1]))
        results = model.predict(crops, imgsz=tile, conf=conf, verbose=False)
        for (ox, oy), res in zip(chunk, results):
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                boxes.append([x1 + ox, y1 + oy, x2 + ox, y2 + oy])
                scores.append(float(b.conf[0]))

    detections = []
    if boxes:
        keep = nms(torch.tensor(boxes), torch.tensor(scores), 0.5).tolist()
        detections = suppress_contained([(boxes[k], scores[k]) for k in keep])
    return img, detections, len(origins)


def save_annotated(img, detections, out_path, max_side=1600):
    scale = min(1.0, max_side / max(img.size))
    small = img.resize((int(img.width * scale), int(img.height * scale)))
    draw = ImageDraw.Draw(small)
    for (x1, y1, x2, y2), score in detections:
        box = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
        draw.rectangle(box, outline=(255, 40, 40), width=3)
        draw.text((box[0] + 3, max(0, box[1] - 12)), f"{score:.2f}", fill=(255, 255, 0))
    small.save(out_path, quality=85)


def main():
    parser = argparse.ArgumentParser(description="Пакетный анализ снимков с дрона")
    parser.add_argument("folder", help="папка со снимками")
    parser.add_argument("--model", default="weeds_model.pt")
    parser.add_argument("--tile", type=int, default=640, help="размер плитки, px")
    parser.add_argument("--overlap", type=float, default=0.2, help="перекрытие плиток, 0..0.5")
    parser.add_argument("--conf", type=float, default=0.25, help="порог уверенности")
    parser.add_argument("--batch", type=int, default=8, help="плиток за один запуск модели")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    images = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in IMG_EXT)
    if not images:
        print(f"В папке {args.folder} нет снимков (jpg, jpeg, jfif, png, webp, bmp, tif, avif).")
        return

    out_dir = Path(args.out)
    (out_dir / "annotated").mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    summary_rows, detection_rows = [], []
    started = time.time()
    for n, path in enumerate(images, 1):
        t0 = time.time()
        try:
            img, detections, n_tiles = analyze_image(
                model, path, args.tile, args.overlap, args.conf, args.batch
            )
        except Exception as e:  # один плохой файл не должен ломать всю папку
            print(f"[{n}/{len(images)}] {path.name}: пропущен ({type(e).__name__}: {e})")
            continue
        save_annotated(img, detections, out_dir / "annotated" / f"{path.stem}.jpg")

        best = max((s for _, s in detections), default=0.0)
        summary_rows.append(
            [path.name, img.width, img.height, n_tiles, len(detections), round(best, 3)]
        )
        for (x1, y1, x2, y2), score in detections:
            detection_rows.append(
                [path.name, round(x1), round(y1), round(x2), round(y2), round(score, 3)]
            )
        print(
            f"[{n}/{len(images)}] {path.name}: {len(detections)} сорняков, "
            f"{n_tiles} плиток, {time.time() - t0:.1f} с"
        )

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "width", "height", "tiles", "weeds", "max_confidence"])
        w.writerows(summary_rows)
    with open(out_dir / "detections.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "x_min", "y_min", "x_max", "y_max", "confidence"])
        w.writerows(detection_rows)

    total = sum(r[4] for r in summary_rows)
    print(f"\nГотово: {len(images)} снимков, {total} сорняков, {time.time() - started:.0f} с.")
    print(f"Результаты в папке: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
