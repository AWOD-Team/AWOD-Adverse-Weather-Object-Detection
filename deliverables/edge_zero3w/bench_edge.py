#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""香橙派 Zero 3W（或任意 ARM/Linux）上的 YOLO11n 推理基准 + 单图检测。

不依赖 torch / ultralytics，只依赖 onnxruntime + numpy + opencv（或 Pillow）。

用法:
  python bench_edge.py --model yolo11n_aw_320_int8.onnx --imgsz 320 --n 30
  python bench_edge.py --model yolo11n_aw_320.onnx --imgsz 320 --n 30
  python detect_edge.py --model yolo11n_aw_320_int8.onnx --image test.jpg --out out.jpg

说明:
  - 类别: 0=vehicle, 1=person
  - ONNX 输出形状 (1, 4+nc, N)，N 为 40*40+20*20+10*10 等网格和
"""
import argparse
import os
import time

import numpy as np
import onnxruntime as ort

CLASSES = ["vehicle", "person"]


def letterbox(img, new, color=(114, 114, 114)):
    import cv2
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new - nh) // 2
    left = (new - nw) // 2
    out = np.full((new, new, 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, (left, top)


def preprocess(img, imgsz):
    lb, r, (l, t) = letterbox(img, imgsz)
    x = np.asarray(lb, dtype=np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    return np.ascontiguousarray(x), r, (l, t)


def nms(boxes, scores, iou_thr=0.45):
    if not len(boxes):
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def postprocess(out, conf_thr, imgsz, scale, pad):
    # out: (1, 4+nc, N) -> (N, 4+nc)
    p = out[0].transpose(1, 0)
    cls_scores = p[:, 4:]
    cls_id = cls_scores.argmax(1)
    conf = cls_scores[np.arange(len(p)), cls_id]
    m = conf >= conf_thr
    p, cls_id, conf = p[m], cls_id[m], conf[m]
    if not len(p):
        return []
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    x1 = (cx - w / 2 - pad[0]) / scale
    y1 = (cy - h / 2 - pad[1]) / scale
    x2 = (cx + w / 2 - pad[0]) / scale
    y2 = (cy + h / 2 - pad[1]) / scale
    boxes = np.stack([x1, y1, x2, y2], 1)
    dets = []
    for c in np.unique(cls_id):
        idx = np.where(cls_id == c)[0]
        for k in nms(boxes[idx], conf[idx]):
            b = boxes[idx][k]
            dets.append((CLASSES[int(c)], float(conf[idx][k]), [float(v) for v in b]))
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--threads", type=int, default=0, help="0=默认")
    ap.add_argument("--image", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    import cv2

    so = ort.SessionOptions()
    if a.threads:
        so.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(a.model, sess_options=so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    print("model:", a.model, "size: %.2f MB" % (os.path.getsize(a.model) / 1e6),
          "input:", sess.get_inputs()[0].shape)

    # 准备输入
    if a.image and os.path.exists(a.image):
        img = cv2.imread(a.image)
    else:
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    x, scale, pad = preprocess(img, a.imgsz)

    for _ in range(3):
        sess.run(None, {iname: x})
    t0 = time.time()
    for _ in range(a.n):
        out = sess.run(None, {iname: x})[0]
    dt = (time.time() - t0) / a.n
    print("inference: %.1f ms/image  (%.1f FPS)  threads=%s" % (dt * 1000, 1 / dt, a.threads or "auto"))

    if a.image and a.out:
        dets = postprocess(out, a.conf, a.imgsz, scale, pad)
        print("detections:", len(dets))
        for name, c, b in dets[:20]:
            print("  %-8s %.3f  [%d,%d,%d,%d]" % (name, c, b[0], b[1], b[2], b[3]))
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            cv2.putText(img, "%s %.2f" % (name, c), (int(b[0]), int(b[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(a.out, img)
        print("saved:", a.out)


if __name__ == "__main__":
    main()
