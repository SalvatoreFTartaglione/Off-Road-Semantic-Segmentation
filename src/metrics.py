"""Evaluation metric: the IoU used by the course, computed in batches on any device.

Definition (identical to the official test notebook):
- for every image and every semantic class c in 1..8 the IoU is
  intersection / union of the binary masks (ground truth == c) and (prediction == c);
- if the class is absent from both the ground truth and the prediction the union is
  empty and the value is NaN, which is ignored;
- if the class is absent from the ground truth but predicted, the value is 0;
- the final score is the NaN-aware mean over the whole (images x classes) matrix.

Label 0 (background) is not scored, but predicting a class on a background pixel still
enlarges the union of that class, so false positives on the background are penalised.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch

from .data import CLASS_NAMES, NUM_CLASSES


def per_image_iou(prediction: torch.Tensor, target: torch.Tensor,
                  num_classes: int = NUM_CLASSES) -> np.ndarray:
    """IoU of every semantic class for every image of a batch.

    Intersection and union are obtained for all images at once by counting, with a
    single bincount, how many pixels of each image have each (predicted / true) label.

    Args:
        prediction: (B, H, W) integer class indices in [0, num_classes).
        target: (B, H, W) integer class indices in [0, num_classes).

    Returns:
        float64 array (B, num_classes - 1) with the IoU of classes 1..num_classes-1
        and NaN where the union is empty.
    """
    if prediction.dim() != 3 or prediction.shape != target.shape:
        raise ValueError(f"expected two (B, H, W) tensors, got {tuple(prediction.shape)} "
                         f"and {tuple(target.shape)}")
    batch = prediction.shape[0]
    size = batch * num_classes
    offset = torch.arange(batch, device=prediction.device).view(-1, 1, 1) * num_classes
    prediction_index = (prediction.long() + offset).reshape(-1)
    target_index = (target.long().to(prediction.device) + offset).reshape(-1)

    prediction_area = torch.bincount(prediction_index, minlength=size)
    target_area = torch.bincount(target_index, minlength=size)
    intersection = torch.bincount(prediction_index[prediction_index == target_index], minlength=size)
    union = prediction_area + target_area - intersection

    iou = torch.full((size,), float("nan"), dtype=torch.float64, device=prediction.device)
    nonempty = union > 0
    iou[nonempty] = intersection[nonempty].double() / union[nonempty].double()
    return iou.reshape(batch, num_classes)[:, 1:].cpu().numpy()


class MeanIoU:
    """Accumulates the per-image IoU over several batches and summarises it.

    Usage::

        metric = MeanIoU()
        for images, masks in loader:
            metric.update(model(images), masks)   # logits (B, C, H, W) or labels (B, H, W)
        results = metric.compute()
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self._scores: list[np.ndarray] = []

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """Add a batch. ``prediction`` may be logits (B, C, H, W) or labels (B, H, W)."""
        if prediction.dim() == 4:
            prediction = prediction.argmax(dim=1)
        self._scores.append(per_image_iou(prediction, target, self.num_classes))

    def compute(self) -> dict[str, np.ndarray | float]:
        """Summarise everything seen since the last reset.

        Returns a dict with:
            score:          the official score, NaN-aware mean over all (image, class) entries.
            mean_class_iou: mean of the per-class IoUs (each class counts the same).
            per_class:      (num_classes - 1,) NaN-aware mean IoU of each class over the images.
            per_image:      (N,) NaN-aware mean IoU of each image over the classes.
            iou_matrix:     (N, num_classes - 1) the raw IoU of every image and class.
        """
        if not self._scores:
            raise RuntimeError("compute() called before any update()")
        iou_matrix = np.concatenate(self._scores, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices give NaN
            per_class = np.nanmean(iou_matrix, axis=0)
            return {
                "score": float(np.nanmean(iou_matrix)),
                "mean_class_iou": float(np.nanmean(per_class)),
                "per_class": per_class,
                "per_image": np.nanmean(iou_matrix, axis=1),
                "iou_matrix": iou_matrix,
            }


def metrics_report(results: dict) -> pd.DataFrame:
    """Per-class table: IoU and in how many images the class counted (non-NaN entries)."""
    return pd.DataFrame({
        "name": CLASS_NAMES[1:],
        "iou": results["per_class"],
        "images": (~np.isnan(results["iou_matrix"])).sum(0),
    }, index=range(1, NUM_CLASSES)).round(4)