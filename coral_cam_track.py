#!/usr/bin/env python3
"""Motion tracking + Edge TPU classification from a USB camera on Coral Dev Board Mini."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate


def load_labels(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if parts[0].isdigit():
                idx = int(parts[0])
                name = parts[1] if len(parts) > 1 else str(idx)
                labels[idx] = name
            else:
                labels[len(labels)] = line
    return labels


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--display-scale", type=float, default=1.0)
    parser.add_argument("--min-area", type=int, default=200)
    parser.add_argument("--history", type=int, default=400)
    parser.add_argument("--var-threshold", type=int, default=9)
    parser.add_argument("--detect-shadows", action="store_true")
    parser.add_argument("--box-pad", type=int, default=12)
    parser.add_argument("--dilate", type=int, default=2)
    parser.add_argument("--merge-iou", type=float, default=0.2)
    parser.add_argument("--track-iou", type=float, default=0.3)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--reid-seconds", type=float, default=3.0)
    parser.add_argument("--inertia", type=float, default=0.7)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"),
    )
    parser.add_argument("--labels", type=Path, default=Path("models/coco_labels.txt"))
    parser.add_argument("--classify-after-seconds", type=float, default=3.0)
    parser.add_argument("--classify-interval", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Unable to open camera: {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=args.history,
        varThreshold=args.var_threshold,
        detectShadows=args.detect_shadows,
    )

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    if not args.labels.exists():
        raise SystemExit(f"Labels not found: {args.labels}")

    labels = load_labels(args.labels)
    allowed_labels = {"person", "car"}

    interpreter = Interpreter(
        model_path=str(args.model),
        experimental_delegates=[load_delegate("libedgetpu.so.1")],
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    wait_ms = max(1, int(1000 / max(1, args.fps))) if not args.no_display else 1

    next_track_id = 1
    tracks = []

    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_h, frame_w = frame.shape[:2]
        mask = subtractor.apply(frame)
        mask = cv2.medianBlur(mask, 5)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        if args.dilate > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=args.dilate)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            if cv2.contourArea(c) < args.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            pad = max(0, args.box_pad)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame_w - 1, x + w + pad)
            y2 = min(frame_h - 1, y + h + pad)
            detections.append((x1, y1, x2, y2))

        # Merge overlapping detections.
        merged = []
        for box in detections:
            merged_into = False
            for i, m in enumerate(merged):
                if args.merge_iou > 0 and iou(box, m) >= args.merge_iou:
                    x1 = min(box[0], m[0])
                    y1 = min(box[1], m[1])
                    x2 = max(box[2], m[2])
                    y2 = max(box[3], m[3])
                    merged[i] = (x1, y1, x2, y2)
                    merged_into = True
                    break
            if not merged_into:
                merged.append(box)
        detections = merged

        now = time.time()
        matched_ids = set()

        for det in detections:
            best_iou = 0.0
            best_track = None
            for tr in tracks:
                overlap = iou(det, tr["bbox"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_track = tr
            if best_track is not None and best_iou >= args.track_iou:
                if args.inertia > 0.0:
                    ox1, oy1, ox2, oy2 = best_track["bbox"]
                    nx1, ny1, nx2, ny2 = det
                    a = max(0.0, min(1.0, args.inertia))
                    bx1 = int(ox1 * a + nx1 * (1.0 - a))
                    by1 = int(oy1 * a + ny1 * (1.0 - a))
                    bx2 = int(ox2 * a + nx2 * (1.0 - a))
                    by2 = int(oy2 * a + ny2 * (1.0 - a))
                    best_track["bbox"] = (bx1, by1, bx2, by2)
                else:
                    best_track["bbox"] = det
                best_track["age"] = 0
                best_track["lost_box"] = None
                best_track["lost_until"] = 0.0
                matched_ids.add(best_track["id"])
            else:
                reused = False
                tracks.sort(key=lambda t: t["id"])
                for tr in tracks:
                    if tr.get("lost_until", 0.0) <= now:
                        continue
                    lb = tr.get("lost_box")
                    if lb is None:
                        continue
                    x1, y1, x2, y2 = lb
                    dx1, dy1, dx2, dy2 = det
                    if dx1 >= x1 and dy1 >= y1 and dx2 <= x2 and dy2 <= y2:
                        if args.inertia > 0.0:
                            ox1, oy1, ox2, oy2 = tr["bbox"]
                            nx1, ny1, nx2, ny2 = det
                            a = max(0.0, min(1.0, args.inertia))
                            bx1 = int(ox1 * a + nx1 * (1.0 - a))
                            by1 = int(oy1 * a + ny1 * (1.0 - a))
                            bx2 = int(ox2 * a + nx2 * (1.0 - a))
                            by2 = int(oy2 * a + ny2 * (1.0 - a))
                            tr["bbox"] = (bx1, by1, bx2, by2)
                        else:
                            tr["bbox"] = det
                        tr["age"] = 0
                        tr["lost_box"] = None
                        tr["lost_until"] = 0.0
                        matched_ids.add(tr["id"])
                        reused = True
                        break
                if not reused:
                    tracks.append(
                        {
                            "id": next_track_id,
                            "bbox": det,
                            "age": 0,
                            "first_seen": now,
                            "last_classified": 0.0,
                            "label": None,
                            "score": None,
                            "lost_box": None,
                            "lost_until": 0.0,
                        }
                    )
                    matched_ids.add(next_track_id)
                    next_track_id += 1

        for tr in tracks:
            if tr["id"] not in matched_ids:
                tr["age"] += 1
                if tr.get("lost_until", 0.0) <= now:
                    x1, y1, x2, y2 = tr["bbox"]
                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w / 2.0
                    cy = y1 + h / 2.0
                    nw = w * 2.0
                    nh = h * 2.0
                    lx1 = max(0, int(cx - nw / 2.0))
                    ly1 = max(0, int(cy - nh / 2.0))
                    lx2 = min(frame_w - 1, int(cx + nw / 2.0))
                    ly2 = min(frame_h - 1, int(cy + nh / 2.0))
                    tr["lost_box"] = (lx1, ly1, lx2, ly2)
                    tr["lost_until"] = now + args.reid_seconds

        tracks = [tr for tr in tracks if tr["age"] <= args.max_age]
        tracks.sort(key=lambda t: t["id"])
        filtered = []
        for tr in tracks:
            if any(iou(tr["bbox"], kept["bbox"]) >= args.track_iou for kept in filtered):
                continue
            filtered.append(tr)
        tracks = filtered

        for tr in tracks:
            if (
                (now - tr["first_seen"]) >= args.classify_after_seconds
                and (now - tr["last_classified"]) >= args.classify_interval
            ):
                x1, y1, x2, y2 = tr["bbox"]
                crop = frame[y1:y2, x1:x2]
                if crop.size != 0:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    ih, iw = input_details["shape"][1:3]
                    resized = cv2.resize(crop_rgb, (iw, ih))
                    if input_details["dtype"] == np.float32:
                        input_tensor = resized.astype(np.float32)
                    else:
                        input_tensor = resized.astype(input_details["dtype"])
                    input_tensor = np.expand_dims(input_tensor, axis=0)
                    interpreter.set_tensor(input_details["index"], input_tensor)
                    interpreter.invoke()
                    class_ids = interpreter.get_tensor(output_details[1]["index"])[0]
                    scores = interpreter.get_tensor(output_details[2]["index"])[0]
                    num = int(interpreter.get_tensor(output_details[3]["index"])[0])
                    best = None
                    for i in range(num):
                        score = float(scores[i])
                        cls_id = int(class_ids[i])
                        label = labels.get(cls_id, str(cls_id))
                        if label not in allowed_labels:
                            continue
                        if best is None or score > best[1]:
                            best = (label, score)
                    if best is not None:
                        label, score = best
                        if tr["score"] is None or score > tr["score"]:
                            tr["label"] = label
                            tr["score"] = score
                    tr["last_classified"] = now

            if tr.get("label") in allowed_labels and tr.get("score") is not None:
                x1, y1, _, _ = tr["bbox"]
                text = f"ID {tr['id']} {tr['label']} {tr['score']:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.5
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
                tx = max(0, x1)
                ty = max(th + baseline + 2, y1 - 5)
                x2 = min(frame_w - 1, tx + tw + 4)
                y1_box = max(0, ty - th - baseline - 2)
                y2_box = min(frame_h - 1, ty + baseline + 2)
                roi = frame[y1_box:y2_box, tx:x2]
                if roi.size != 0:
                    frame[y1_box:y2_box, tx:x2] = 255 - roi
                cv2.putText(
                    frame,
                    text,
                    (tx + 2, ty),
                    font,
                    scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )

        if not args.no_display:
            display_frame = frame
            if args.display_scale != 1.0:
                display_frame = cv2.resize(
                    frame,
                    (int(frame_w * args.display_scale), int(frame_h * args.display_scale)),
                )
            cv2.imshow("Coral Motion Track", display_frame)
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == 27 or key == ord("q"):
                break

        if args.max_seconds and (now - start_time) >= args.max_seconds:
            break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
