from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

import numpy as np

from core.dehaze.hybrid_dehaze import HybridDehazer
from core.detection.yolo_detector import YOLODetector


class AWODPipeline:
    def __init__(self, yolo_model_path: str, dehazer: HybridDehazer | None = None,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 enable_dehaze: bool = True):
        self.detector = YOLODetector(yolo_model_path, conf_threshold, iou_threshold)
        self.dehazer = dehazer or HybridDehazer()
        self.enable_dehaze = enable_dehaze

    def process(self, img: np.ndarray) -> dict:
        timing = {}
        if self.enable_dehaze:
            t0 = time.perf_counter()
            enhanced = self.dehazer.dehaze(img)
            t1 = time.perf_counter()
            timing["dehaze_ms"] = (t1 - t0) * 1000
        else:
            enhanced = img
            timing["dehaze_ms"] = 0.0

        t0 = time.perf_counter()
        detections = self.detector.detect(enhanced)
        t1 = time.perf_counter()
        timing["detect_ms"] = (t1 - t0) * 1000

        return {
            "detections": detections,
            "enhanced_img": enhanced,
            "timing": timing,
        }

    def process_and_draw(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        result = self.process(img)
        raw_drawn = self.detector.draw(img, result["detections"])
        enhanced_drawn = self.detector.draw(result["enhanced_img"], result["detections"])
        return raw_drawn, enhanced_drawn, result["timing"]
