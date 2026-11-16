"""Prediction and evaluation on a test set.

``predict`` is the function required by the project: it receives raw images (uint8, any size) and
returns the label maps (uint8, same size), doing every pre- and post-processing step itself::

    from src.predict import load_model, predict
    model = load_model()                      # the network of config.FINAL_WEIGHTS
    labels = predict(model, image)            # image (H, W, 3)    -> labels (H, W)
    labels = predict(model, batch)            # batch (B, H, W, 3) -> labels (B, H, W, 1)
    labels = predict(batch)                   # the same, with the model loaded the first time it is needed

Pipeline for every image:

1. resize to the training resolution (``area`` interpolation, as in training) and normalise with the
   mean and standard deviation stored in the checkpoint;
2. forward pass, in float16 on the GPU, in chunks of ``config.PREDICT_BATCH_SIZE`` images;
3. the class scores are enlarged to the original resolution (bilinear) and the best class of every
   pixel is chosen;
4. post-processing: a class that covers less than ``min_area`` of the image is removed from it and its
   pixels go to the best remaining class. The official metric gives IoU 0 to a class that is absent
   from the ground truth but predicted even on a few pixels, and the validation set showed that this
   was a large part of the lost score.

``evaluate`` computes the official metric on a folder of samples, exactly like the evaluation code of
the course (one image at a time, mean over all the (image, class) pairs ignoring NaN).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from . import config
from .data import CLASS_NAMES, NUM_CLASSES, colorize, download_dataset, get_palette, load_sample
from .train import load_checkpoint


# --------------------------------------------------------------------------- #
# The model and its prediction settings
# --------------------------------------------------------------------------- #
@dataclass
class Segmenter:
    """A trained network together with everything ``predict`` needs to use it.

    Attributes:
        model: the network in evaluation mode, on ``device``.
        mean, std: normalisation of the input (RGB, in [0, 1]), the ones used in training.
        image_size: (height, width) the network works at.
        min_area, rare_min_area, rare_classes: post-processing (see ``suppress_small_classes``).
        batch_size: images per forward pass.
    """
    model: torch.nn.Module
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    image_size: tuple[int, int]
    device: torch.device
    min_area: float = config.POSTPROCESS_MIN_AREA
    rare_min_area: float = config.POSTPROCESS_RARE_MIN_AREA
    rare_classes: tuple[int, ...] = config.RARE_CLASSES
    batch_size: int = config.PREDICT_BATCH_SIZE

    def without_postprocessing(self) -> "Segmenter":
        """The same network with the post-processing switched off (to measure what it adds)."""
        return replace(self, min_area=0.0, rare_min_area=0.0)


def load_model(path: str | Path | None = None, device: str | None = None, **settings) -> Segmenter:
    """Load a trained network from a checkpoint (default: ``config.FINAL_WEIGHTS``).

    Args:
        device: "cuda" or "cpu". Default: the GPU if there is one.
        settings: any field of ``Segmenter`` to change (``min_area``, ``batch_size``, ...).
    """
    path = Path(path) if path else config.FINAL_WEIGHTS
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, state = load_checkpoint(path, device)
    model = model.to(memory_format=torch.channels_last)
    return Segmenter(model=model, mean=tuple(state["mean"]), std=tuple(state["std"]),
                     image_size=tuple(state["image_size"]), device=device, **settings)


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def suppress_small_classes(scores: torch.Tensor, min_area: float, rare_min_area: float = 0.0,
                           rare_classes: tuple[int, ...] = config.RARE_CLASSES) -> torch.Tensor:
    """Label every pixel with its best class, after removing the classes that cover too little.

    Args:
        scores: (B, classes, H, W) class scores (logits or probabilities).
        min_area: a class whose predicted area is below this fraction of the image is removed from
            that image. The background (class 0) is never removed.
        rare_min_area: the same threshold for ``rare_classes``.

    Returns:
        (B, H, W) int64 labels. The areas are measured on the plain argmax of the original scores.
    """
    labels = scores.argmax(1)
    if min_area <= 0 and rare_min_area <= 0:
        return labels
    pixels = labels.shape[-1] * labels.shape[-2]
    scores = scores.clone()
    for c in range(1, scores.shape[1]):
        threshold = (rare_min_area if c in rare_classes else min_area) * pixels
        area = (labels == c).flatten(1).sum(1)
        small = (area > 0) & (area < threshold)
        scores[small, c] = float("-inf")
    return scores.argmax(1)


def _preprocess(images: np.ndarray, segmenter: Segmenter) -> torch.Tensor:
    """(B, H, W, 3) uint8 -> (B, 3, h, w) normalised float32 tensor at the working resolution."""
    height, width = segmenter.image_size
    resized = np.stack([cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) for image in images])
    batch = torch.from_numpy(resized).to(segmenter.device).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(segmenter.mean, device=segmenter.device).view(1, 3, 1, 1)
    std = torch.tensor(segmenter.std, device=segmenter.device).view(1, 3, 1, 1)
    return ((batch - mean) / std).contiguous(memory_format=torch.channels_last)


_DEFAULT_SEGMENTER: Segmenter | None = None


@torch.no_grad()
def predict(model, X=None) -> np.ndarray:
    """Predict the label map of an image or of a batch of images.

    Args:
        model: a ``Segmenter`` (see ``load_model``). It may be omitted, ``predict(X)``: the network
            of ``config.FINAL_WEIGHTS`` is then loaded the first time and reused.
        X: uint8 array, either one image (H, W, 3) or a batch (B, H, W, 3), of any size and any B.

    Returns:
        uint8 array with the class of every pixel (0 = background, 1..8): (H, W) for one image,
        (B, H, W, 1) for a batch, like the label files.
    """
    global _DEFAULT_SEGMENTER
    if X is None:                                   # predict(X)
        X = model
        if _DEFAULT_SEGMENTER is None:
            _DEFAULT_SEGMENTER = load_model()
        model = _DEFAULT_SEGMENTER
    X = np.asarray(X)
    if X.dtype != np.uint8 or X.ndim not in (3, 4) or X.shape[-1] != 3:
        raise ValueError(f"expected a uint8 array (H, W, 3) or (B, H, W, 3), got {X.dtype} {X.shape}")
    single = X.ndim == 3
    batch = X[None] if single else X
    height, width = batch.shape[1:3]

    outputs = []
    for start in range(0, len(batch), model.batch_size):
        chunk = _preprocess(batch[start:start + model.batch_size], model)
        with torch.autocast(device_type=model.device.type, dtype=torch.float16, enabled=model.device.type == "cuda"):
            logits = model.model(chunk)
        logits = F.interpolate(logits.float(), size=(height, width), mode="bilinear", align_corners=False)
        labels = suppress_small_classes(logits, model.min_area, model.rare_min_area, model.rare_classes)
        outputs.append(labels.to(torch.uint8).cpu().numpy())
    labels = np.concatenate(outputs) if outputs else np.zeros((0, height, width), dtype=np.uint8)
    return labels[0] if single else labels[..., None]


# --------------------------------------------------------------------------- #
# Evaluation (the metric of the course)
# --------------------------------------------------------------------------- #
def compute_iou(mask1: np.ndarray, mask2: np.ndarray, label: int) -> float:
    """IoU of one class between two label maps, NaN if the class is in neither of them."""
    intersection = np.sum((mask1 == label) & (mask2 == label))
    union = np.sum((mask1 == label) | (mask2 == label))
    if union == 0:
        return np.nan
    return intersection / union


def compute_all_iou(mask1: np.ndarray, mask2: np.ndarray, num_labels: int = NUM_CLASSES - 1) -> np.ndarray:
    """IoU of the classes 1..8 (the background is skipped): an array of 8 values, possibly NaN."""
    return np.array([compute_iou(mask1, mask2, label + 1) for label in range(num_labels)])


@dataclass
class Evaluation:
    """Result of ``evaluate``.

    Attributes:
        iou: (images, 8) IoU of every class in every image, NaN where the class is in neither map.
        samples: the sample folders, in the order of the rows of ``iou``.
        score: the official score, the mean of ``iou`` ignoring NaN.
        seconds_per_image: time of the whole ``predict`` call per image (preprocessing included).
        peak_gpu_gb: peak GPU memory allocated by PyTorch during the evaluation (0 on the CPU).
    """
    iou: np.ndarray
    samples: list[Path]
    score: float
    seconds_per_image: float
    peak_gpu_gb: float

    @property
    def per_class(self) -> np.ndarray:
        """Mean IoU of every class over the images where it is valid."""
        return np.nanmean(self.iou, axis=0)

    @property
    def per_image(self) -> np.ndarray:
        """Mean IoU of every image over its valid classes."""
        return np.nanmean(self.iou, axis=1)

    @property
    def fps(self) -> float:
        return 1.0 / self.seconds_per_image

    def table(self):
        """Per-class summary as a DataFrame: mean IoU, images where the class counts, entries equal to 0."""
        import pandas as pd
        valid = ~np.isnan(self.iou)
        return pd.DataFrame({"mean IoU": self.per_class, "valid images": valid.sum(0),
                             "IoU = 0": (self.iou == 0).sum(0)}, index=CLASS_NAMES[1:]).round(4)


def evaluate(segmenter: Segmenter, test_dir: str | Path = config.TEST_DIR, limit: int | None = None,
             verbose: bool = True) -> Evaluation:
    """Official score of a network on a folder of samples (rgb.jpg + labels.png), at full resolution.

    Like the evaluation code of the course: the images are given to ``predict`` one at a time, at
    their original size, and the IoU of the classes 1..8 is computed against the label file.
    """
    samples = download_dataset(test_dir)
    if limit:
        samples = samples[:limit]
    if not samples:
        raise FileNotFoundError(f"no samples (rgb.jpg + labels.png) found in {test_dir}")
    ious = np.zeros((len(samples), NUM_CLASSES - 1))
    elapsed = 0.0
    on_gpu = segmenter.device.type == "cuda"
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    predict(segmenter, load_sample(samples[0])[0])                     # warm-up: the first call is slow
    for i, sample in enumerate(samples):
        image, labels = load_sample(sample)
        if on_gpu:
            torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = predict(segmenter, image)
        if on_gpu:
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        ious[i] = compute_all_iou(labels, prediction)
    result = Evaluation(iou=ious, samples=samples, score=float(np.nanmean(ious)),
                        seconds_per_image=elapsed / len(samples),
                        peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if on_gpu else 0.0)
    if verbose:
        print(f"{len(samples)} samples | score {result.score:.4f} | {result.fps:.1f} fps | "
              f"peak GPU memory {result.peak_gpu_gb:.2f} GB")
    return result


def compare_postprocessing(segmenter: Segmenter, test_dir: str | Path = config.TEST_DIR) -> dict[str, Evaluation]:
    """Evaluate the network without and with the post-processing, to show what the latter adds."""
    return {"plain": evaluate(segmenter.without_postprocessing(), test_dir),
            "post-processed": evaluate(segmenter, test_dir)}


def plot_predictions(segmenter: Segmenter, samples: list[Path], title: str = ""):
    """Image, ground truth and prediction side by side, one row per sample."""
    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 2.4 * len(samples)), squeeze=False)
    for row, sample in zip(axes, samples):
        image, labels = load_sample(sample)
        palette = get_palette(sample)
        prediction = predict(segmenter, image)
        for ax, content, name in zip(row, (image, colorize(labels, palette), colorize(prediction, palette)),
                                     ("image", "ground truth", "prediction")):
            ax.imshow(content)
            ax.set_title(f"{sample.name}: {name}", fontsize=9)
            ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
