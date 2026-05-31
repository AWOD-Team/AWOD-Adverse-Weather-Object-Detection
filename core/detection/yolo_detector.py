from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, img: np.ndarray) -> list[dict]:
        results = self.model(img, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            names = result.names
            for box, cls, conf in zip(boxes, classes, confs):
                detections.append({
                    "bbox": box.tolist(),
                    "class_id": int(cls),
                    "class_name": names.get(int(cls), str(cls)),
                    "confidence": float(conf),
                })
        return detections

    def draw(self, img: np.ndarray, detections: list[dict]) -> np.ndarray:
        img_out = img.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_out, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(img_out, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return img_out
