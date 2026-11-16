"""Leakage-aware train/validation split.

Frames come from a moving camera, so frames with close IDs are near duplicates.
A random split would put near duplicates in both sets and inflate the validation
score. This module builds contiguous blocks of frames, assigns whole blocks to
train or validation while balancing the class distribution, and finally purges
the training frames that sit right next to a validation block.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.data import CLASS_NAMES, NUM_CLASSES


@dataclass
class SplitResult:
    """Outcome of a split. All arrays are indexed like the sorted sample list."""

    sample_names: list[str]
    block_ids: np.ndarray
    is_val: np.ndarray
    is_purged: np.ndarray
    params: dict

    @property
    def train_indices(self) -> np.ndarray:
        return np.flatnonzero(~self.is_val & ~self.is_purged)

    @property
    def val_indices(self) -> np.ndarray:
        return np.flatnonzero(self.is_val)

    @property
    def purged_indices(self) -> np.ndarray:
        return np.flatnonzero(self.is_purged)


def build_blocks(consecutive: np.ndarray, min_block_size: int = 30,
                 max_block_size: int = 120, cut_quantile: float = 0.5) -> np.ndarray:
    """Group frame indices into contiguous blocks, cutting only at scene changes.

    A block is closed once it has at least ``min_block_size`` frames AND the
    difference between the current frame and the next one is above the
    ``cut_quantile`` quantile of all consecutive differences (a scene change).
    Runs of near-duplicate frames are never cut, unless a block reaches
    ``max_block_size`` frames.

    Args:
        consecutive: array of length n-1, difference between frame i and i+1.

    Returns:
        Integer array of length n with the block id of every frame.
    """
    n = len(consecutive) + 1
    threshold = np.quantile(consecutive, cut_quantile)
    block_ids = np.zeros(n, dtype=np.int64)
    current, size = 0, 1
    for i in range(1, n):
        can_cut = size >= min_block_size and consecutive[i - 1] >= threshold
        must_cut = size >= max_block_size
        if can_cut or must_cut:
            current += 1
            size = 0
        block_ids[i] = current
        size += 1

    # A tiny trailing block would be hard to balance: merge it into the previous one
    if current > 0 and size < min_block_size // 2:
        block_ids[block_ids == current] = current - 1
    return block_ids


def block_statistics(per_image: np.ndarray, block_ids: np.ndarray):
    """Aggregate per-image class counts into per-block statistics.

    Returns:
        pixels: (n_blocks, 9) pixels of each class in each block.
        images: (n_blocks, 9) images of each block containing each class.
        frames: (n_blocks,) number of frames in each block.
    """
    n_blocks = int(block_ids.max()) + 1
    pixels = np.zeros((n_blocks, NUM_CLASSES), dtype=np.int64)
    images = np.zeros((n_blocks, NUM_CLASSES), dtype=np.int64)
    frames = np.zeros(n_blocks, dtype=np.int64)
    np.add.at(pixels, block_ids, per_image)
    np.add.at(images, block_ids, (per_image > 0).astype(np.int64))
    np.add.at(frames, block_ids, 1)
    return pixels, images, frames


def assign_blocks(pixels: np.ndarray, images: np.ndarray, frames: np.ndarray,
                  val_fraction: float = 0.2, n_trials: int = 300, seed: int = 42) -> np.ndarray:
    """Choose which blocks go to validation so that class shares match val_fraction.

    The cost is the mean relative deviation, over classes 1..8, of the validation
    share of (a) pixels and (b) images containing the class from the target
    fraction, plus the relative deviation of the share of frames. Each trial
    starts from a random assignment and greedily swaps one validation block with
    one training block while the cost decreases. The best trial is returned.

    Returns:
        Boolean array of length n_blocks, True for validation blocks.
    """
    rng = np.random.default_rng(seed)
    n_blocks = len(frames)
    total_pixels = np.maximum(pixels.sum(0)[1:], 1)
    total_images = np.maximum(images.sum(0)[1:], 1)
    total_frames = frames.sum()

    def cost(flags: np.ndarray) -> float:
        pixel_share = pixels[flags].sum(0)[1:] / total_pixels
        image_share = images[flags].sum(0)[1:] / total_images
        frame_share = frames[flags].sum() / total_frames
        value = (np.abs(pixel_share - val_fraction).mean() / val_fraction
                 + np.abs(image_share - val_fraction).mean() / val_fraction
                 + abs(frame_share - val_fraction) / val_fraction)
        # Every evaluated class must appear in the validation images
        if (images[flags].sum(0)[1:] == 0).any():
            value += 10.0
        return float(value)

    best_flags, best_cost = None, np.inf
    for _ in range(n_trials):
        flags = np.zeros(n_blocks, dtype=bool)
        accumulated = 0
        for block in rng.permutation(n_blocks):
            if accumulated >= val_fraction * total_frames:
                break
            flags[block] = True
            accumulated += frames[block]
        current = cost(flags)

        improved = True
        while improved:
            improved = False
            for i in np.flatnonzero(flags):
                for j in np.flatnonzero(~flags):
                    candidate = flags.copy()
                    candidate[i], candidate[j] = False, True
                    candidate_cost = cost(candidate)
                    if candidate_cost < current - 1e-12:
                        flags, current = candidate, candidate_cost
                        improved = True
                        break
                if improved:
                    break

        if current < best_cost:
            best_flags, best_cost = flags, current
    return best_flags


def purge_neighbours(is_val: np.ndarray, gap: int) -> np.ndarray:
    """Flag the non-validation frames closer than ``gap`` IDs to a validation frame."""
    if gap <= 0:
        return np.zeros_like(is_val)
    kernel = np.ones(2 * gap + 1)
    near_val = np.convolve(is_val.astype(float), kernel, mode="same") > 0
    return near_val & ~is_val


def make_split(sample_names: list[str], per_image: np.ndarray, consecutive: np.ndarray,
               val_fraction: float = 0.2, min_block_size: int = 30, max_block_size: int = 120,
               cut_quantile: float = 0.5, purge_gap: int = 5, n_trials: int = 300,
               seed: int = 42) -> SplitResult:
    """Build a block-based, class-balanced, leakage-purged train/validation split.

    Args:
        sample_names: sample directory names, in the same order as ``per_image``.
        per_image: (n, 9) pixels of each class in each image (from the dataset stats).
        consecutive: (n-1,) difference between frame i and i+1 (from consecutive_similarity).
        purge_gap: training frames within this many IDs of a validation frame are
            excluded from training (5 is the decorrelation lag measured on the data).
    """
    if not (len(sample_names) == len(per_image) == len(consecutive) + 1):
        raise ValueError("sample_names, per_image and consecutive have inconsistent lengths")

    block_ids = build_blocks(consecutive, min_block_size, max_block_size, cut_quantile)
    pixels, images, frames = block_statistics(per_image, block_ids)
    block_is_val = assign_blocks(pixels, images, frames, val_fraction, n_trials, seed)
    is_val = block_is_val[block_ids]
    is_purged = purge_neighbours(is_val, purge_gap)

    params = {
        "val_fraction": val_fraction, "min_block_size": min_block_size,
        "max_block_size": max_block_size, "cut_quantile": cut_quantile,
        "purge_gap": purge_gap, "n_trials": n_trials, "seed": seed,
    }
    return SplitResult(list(sample_names), block_ids, is_val, is_purged, params)


def split_report(split: SplitResult, per_image: np.ndarray) -> pd.DataFrame:
    """Compare class distribution between training and validation (classes 1..8)."""
    train, val = split.train_indices, split.val_indices
    train_pixels = per_image[train, 1:].sum(0)
    val_pixels = per_image[val, 1:].sum(0)
    return pd.DataFrame({
        "name": CLASS_NAMES[1:],
        "train_pixel_%": train_pixels / train_pixels.sum() * 100,
        "val_pixel_%": val_pixels / val_pixels.sum() * 100,
        "train_images": (per_image[train, 1:] > 0).sum(0),
        "val_images": (per_image[val, 1:] > 0).sum(0),
    }, index=range(1, NUM_CLASSES)).round(2)


def plot_split(split: SplitResult, per_image: np.ndarray):
    """Show where train/val/purged frames lie along the ID order and compare classes."""
    codes = np.zeros(len(split.sample_names), dtype=int)
    codes[split.is_val] = 1
    codes[split.is_purged] = 2
    colors = ["#4c72b0", "#dd8452", "#bbbbbb"]

    fig = plt.figure(figsize=(15, 7))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 3])

    strip = fig.add_subplot(grid[0, :])
    strip.imshow(codes[None, :], aspect="auto", cmap=ListedColormap(colors),
                 vmin=0, vmax=2, interpolation="nearest")
    for edge in np.flatnonzero(np.diff(split.block_ids)) + 0.5:
        strip.axvline(edge, color="white", lw=0.6)
    strip.set_yticks([])
    strip.set_xlabel("frame index (white lines = block boundaries)")
    strip.set_title("Split along the ID order")
    strip.legend(handles=[Patch(color=c, label=l) for c, l in
                          zip(colors, ["train", "validation", "purged (unused)"])],
                 loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=3)

    report = split_report(split, per_image)
    x = np.arange(len(report))
    n_train, n_val = len(split.train_indices), len(split.val_indices)

    ax_pixels = fig.add_subplot(grid[1, 0])
    ax_pixels.bar(x - 0.2, report["train_pixel_%"], 0.4, label="train", color=colors[0])
    ax_pixels.bar(x + 0.2, report["val_pixel_%"], 0.4, label="validation", color=colors[1])
    ax_pixels.set_yscale("log")
    ax_pixels.set_ylabel("share of pixels (%)")
    ax_pixels.set_title("Class distribution (pixels)")

    ax_images = fig.add_subplot(grid[1, 1])
    ax_images.bar(x - 0.2, report["train_images"] / n_train * 100, 0.4, label="train", color=colors[0])
    ax_images.bar(x + 0.2, report["val_images"] / n_val * 100, 0.4, label="validation", color=colors[1])
    ax_images.set_ylabel("images containing the class (%)")
    ax_images.set_title("Class presence (images)")

    for ax in (ax_pixels, ax_images):
        ax.set_xticks(x)
        ax.set_xticklabels(report["name"], rotation=45, ha="right")
        ax.legend()
    fig.tight_layout()
    return fig


def save_split(split: SplitResult, path: str | Path) -> None:
    """Write the split protocol (parameters and sample names) to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = split.sample_names
    payload = {
        "params": split.params,
        "n_train": int(len(split.train_indices)),
        "n_val": int(len(split.val_indices)),
        "n_purged": int(len(split.purged_indices)),
        "train": [names[i] for i in split.train_indices],
        "val": [names[i] for i in split.val_indices],
        "purged": [names[i] for i in split.purged_indices],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Split saved to {path}")


def load_split(path: str | Path, samples: list[Path]) -> tuple[list[Path], list[Path]]:
    """Load a saved split and return (train_samples, val_samples) as directory paths."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_name = {p.name: p for p in samples}
    if len(by_name) != len(samples):
        # Two samples with the same name (e.g. in data/train and data/test) would silently replace each other
        raise ValueError("sample names are not unique: pass only the samples of one folder")
    missing = [n for n in payload["train"] + payload["val"] if n not in by_name]
    if missing:
        raise ValueError(f"{len(missing)} samples of the split are not in the dataset, e.g. {missing[:3]}")
    return [by_name[n] for n in payload["train"]], [by_name[n] for n in payload["val"]]