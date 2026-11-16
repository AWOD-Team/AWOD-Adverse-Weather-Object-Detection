"""AWOD APIs. Image preprocessing can run without importing Torch/YOLO."""

from core.dehaze import DarkChannelPrior, Retinex, HybridDehazer, RetinexBDCP

__all__ = ["AWODPipeline", "YOLODetector", "DarkChannelPrior", "Retinex",
           "HybridDehazer", "RetinexBDCP"]


def __getattr__(name):
    if name == "AWODPipeline":
        from core.pipeline import AWODPipeline
        return AWODPipeline
    if name == "YOLODetector":
        from core.detection import YOLODetector
        return YOLODetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
