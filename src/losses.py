"""Loss functions for semantic segmentation.

The metric averages the IoU over the 8 semantic classes with the same weight, while
the pixels are extremely unbalanced (high vegetation ~37%, puddle ~0.2%). A plain
cross-entropy is dominated by the frequent classes, so the loss combines:

- a class-weighted cross-entropy, which gives stable gradients for every pixel and
  amplifies the rare classes;
- a Dice loss, which measures the overlap of each class independently of its size
  and is closely related to the IoU used by the metric.

All hyperparameters are read from ``config``.
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm
from . import config
from .data import NUM_CLASSES, load_sample


def count_class_pixels(samples: list[Path], image_size: config.ImageSize | None = config.DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Count the pixels of each class (0..8) over the given samples.

    Call it with the TRAINING samples only. The masks are resized with nearest
    neighbour exactly like in the dataset, so the counts match what the network sees.
    """
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for sample in tqdm(samples, desc="Counting class pixels"):
        _, mask = load_sample(sample)
        if image_size is not None:
            height, width = image_size
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        counts += np.bincount(mask.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    return counts


def compute_class_weights(pixel_counts: np.ndarray, power: float = config.CLASS_WEIGHT_POWER,
                        background_weight: float = config.BACKGROUND_WEIGHT,
                        max_weight: float = config.MAX_CLASS_WEIGHT) -> torch.Tensor:
    """Class weights w_c = (f_max / f_c) ** power for the 8 semantic classes.

    f_c is the pixel frequency of class c and f_max the frequency of the most frequent
    semantic class, so that class gets weight 1 and every rarer class a weight above 1.

    - power = 0 gives uniform weights, power = 1 the full inverse frequency
      (too aggressive for very rare classes), power = 0.5 is a compromise.
    - ``max_weight`` caps the weights, so an extremely rare (or absent) class cannot
      make the training unstable.
    - Class 0 (background) is not part of the metric: it gets the fixed
      ``background_weight``, on the same scale (1.0 = as the most frequent class).
      The cross-entropy divides by the sum of the weights, so the overall magnitude of
      the loss does not depend on these values.

    Returns:
        float32 tensor of shape (9,).
    """
    counts = np.asarray(pixel_counts, dtype=np.float64)
    frequency = np.maximum(counts / counts.sum(), 1e-8)
    semantic = frequency[1:]
    weights = np.empty(NUM_CLASSES, dtype=np.float64)
    weights[1:] = np.minimum((semantic.max() / semantic) ** power, max_weight)
    weights[0] = background_weight
    return torch.tensor(weights, dtype=torch.float32)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = config.DICE_SMOOTH, ignore_background: bool = config.IGNORE_BACKGROUND) -> torch.Tensor:
    """Soft Dice loss averaged over the semantic classes 1..8.

    For every class c: dice_c = (2 * sum(p_c * g_c) + smooth) / (sum(p_c) + sum(g_c) + smooth),
    where p_c are the softmax probabilities and g_c the ground-truth indicator. Sums run
    over the whole batch. The loss is 1 - mean(dice_c). Predicting class c where the
    ground truth is background still increases sum(p_c), so false positives are penalised
    exactly as in the evaluation metric.

    The ground truth is never expanded to a one-hot tensor: the intersection is taken
    with gather/scatter, which saves a lot of memory on large batches.

    Args:
        logits: (B, C, H, W) raw scores.
        target: (B, H, W) integer class indices.
        ignore_background: if True the pixels whose label is 0 do not contribute at all.
    """
    num_classes = logits.shape[1]
    probs = logits.float().softmax(dim=1)
    flat_target = target.reshape(-1)
    # Probability assigned to the true class of every pixel
    true_prob = probs.gather(1, target.unsqueeze(1)).reshape(-1)

    if ignore_background:
        valid = (target != 0)
        probs = probs * valid.unsqueeze(1)
        true_prob = true_prob * valid.reshape(-1)
        target_counts = torch.bincount(flat_target[valid.reshape(-1)], minlength=num_classes)
    else:
        target_counts = torch.bincount(flat_target, minlength=num_classes)

    intersection = torch.zeros(num_classes, device=logits.device, dtype=probs.dtype)
    intersection.scatter_add_(0, flat_target, true_prob)
    cardinality = probs.sum(dim=(0, 2, 3)) + target_counts.to(probs.dtype)

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice[1:].mean()


class SegmentationLoss(nn.Module):
    """Weighted cross-entropy + Dice loss.

    ``loss = ce_weight * CE(class weights) + dice_weight * Dice``

    Args:
        class_weights: (9,) tensor, see ``compute_class_weights``. It is registered as a
            buffer, so ``loss.to(device)`` moves it together with the model.
        ce_weight, dice_weight, dice_smooth, ignore_background: default to ``config``.
    """

    def __init__(self, class_weights: torch.Tensor, ce_weight: float = config.LOSS_CE_WEIGHT,
                dice_weight: float = config.LOSS_DICE_WEIGHT, dice_smooth: float = config.DICE_SMOOTH,
                ignore_background: bool = config.IGNORE_BACKGROUND):
        super().__init__()
        self.register_buffer("class_weights", class_weights.float())
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_smooth = dice_smooth
        self.ignore_background = ignore_background

    def components(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the individual terms and their weighted sum (for logging)."""
        if logits.dim() != 4 or target.dim() != 3:
            raise ValueError(f"expected logits (B, C, H, W) and target (B, H, W), "
                            f"got {tuple(logits.shape)} and {tuple(target.shape)}")
        zero = logits.new_zeros((), dtype=torch.float32)
        ce = zero
        dice = zero
        if self.ce_weight > 0:
            ce = F.cross_entropy(logits.float(), target, weight=self.class_weights,
                                ignore_index=0 if self.ignore_background else -100)
        if self.dice_weight > 0:
            dice = dice_loss(logits, target, self.dice_smooth, self.ignore_background)
        total = self.ce_weight * ce + self.dice_weight * dice
        return {"total": total, "ce": ce, "dice": dice}

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.components(logits, target)["total"]


def create_loss(train_samples: list[Path], image_size: config.ImageSize | None = config.DEFAULT_IMAGE_SIZE,
                device: str | torch.device = "cpu") -> SegmentationLoss:
    """Build the loss with class weights computed on the training samples."""
    weights = compute_class_weights(count_class_pixels(train_samples, image_size))
    return SegmentationLoss(weights).to(device)