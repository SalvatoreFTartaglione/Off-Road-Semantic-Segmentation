"""Data utilities: download, discovery, loading and exploratory analysis.

Everything here is independent from PyTorch on purpose: it only needs
numpy / PIL / matplotlib, so it can be used in the exploration notebook
before any model exists. The torch ``Dataset`` will live in ``dataset.py``.
"""
from __future__ import annotations
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
try:  
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kwargs):
        return it

NUM_CLASSES = 9  # label 0 = background, 1..8 = the eight semantic classes
RGB_NAME = "rgb.jpg"
MASK_NAME = "labels.png"

CLASS_NAMES = [
    "background",                      # 0
    "smooth trail",                    # 1  (da verificare)
    "traversable grass",               # 2
    "rough trail",                     # 3  (da verificare)
    "puddle",                          # 4  (confermato)
    "obstacle",                        # 5
    "non-traversable low vegetation",  # 6
    "high vegetation",                 # 7
    "sky",                             # 8
]


# --------------------------------------------------------------------------- #
# Download / discovery
# --------------------------------------------------------------------------- #

def download_dataset(root: str | Path) -> list[Path]:
    """Return the sorted sample directories under ``root`` (searched recursively).
    A sample directory is any folder containing both ``rgb.jpg`` and
    ``labels.png``. Stray files such as ``.DS_Store`` are ignored.
    """
    root = Path(root)
    if not root.exists():
        print(f"{root} does not exist")
        return []
    return sorted(p.parent for p in root.rglob(RGB_NAME) if (p.parent / MASK_NAME).exists())


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_sample(sample_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one sample as ``(image, mask)``.
    image: (H, W, 3) uint8.   mask: (H, W) uint8 with class indices 0..8.
    The PNG is stored in palette mode: ``np.asarray`` gives the class indices
    directly. Never call ``.convert("RGB")`` on it or the indices are lost.
    """
    sample_dir = Path(sample_dir)
    with Image.open(sample_dir / RGB_NAME) as im:
        image = np.asarray(im.convert("RGB")).copy()
    with Image.open(sample_dir / MASK_NAME) as lb:
        mask = np.asarray(lb).copy()
    return image, mask


def get_palette(sample_dir: str | Path) -> np.ndarray:
    """Return the (256, 3) uint8 colour palette stored in a label PNG."""
    with Image.open(Path(sample_dir) / MASK_NAME) as lb:
        pal = lb.getpalette()
    return np.array(pal, dtype=np.uint8).reshape(-1, 3)


def colorize(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Turn an (H, W) index mask into an (H, W, 3) RGB image."""
    return palette[mask]


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def compute_dataset_stats(samples: list[Path]) -> dict[str, np.ndarray]:
    """One pass over the dataset computing everything we need.

    Returns a dict with
      pixel_counts  (9,)    total pixels per class
      image_counts  (9,)    in how many images each class appears
      per_image     (N, 9)  pixels of each class in each image (used later
                            for a stratified train/val split)
      rgb_mean      (3,)    per-channel mean in [0, 1]
      rgb_std       (3,)    per-channel std in [0, 1]
      shapes        (N, 2)  (H, W) of every image, to check they are uniform
    """
    n = len(samples)
    per_image = np.zeros((n, NUM_CLASSES), dtype=np.int64)
    shapes = np.zeros((n, 2), dtype=np.int64)
    brightness = np.zeros(n)
    ch_sum = np.zeros(3, dtype=np.float64)
    ch_sq = np.zeros(3, dtype=np.float64)
    n_pix = 0

    for i, s in enumerate(tqdm(samples, desc="Analysing dataset")):
        image, mask = load_sample(s)
        per_image[i] = np.bincount(mask.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
        shapes[i] = mask.shape
        x = image.reshape(-1, 3).astype(np.float64) / 255.0
        brightness[i] = x.mean()
        ch_sum += x.sum(0)
        ch_sq += (x ** 2).sum(0)
        n_pix += x.shape[0]

    mean = ch_sum / n_pix
    std = np.sqrt(ch_sq / n_pix - mean ** 2)
    return {
        "pixel_counts": per_image.sum(0),
        "image_counts": (per_image > 0).sum(0),
        "per_image": per_image,
        "brightness": brightness,
        "rgb_mean": mean,
        "rgb_std": std,
        "shapes": shapes,
    }


def plot_leakage_analysis(consecutive, random_, thumbs, max_lag=60, tolerance=0.1):
    """Plot three views of frame similarity and estimate the decorrelation lag.

    The decorrelation lag is the smallest ID distance k whose mean frame
    difference is within ``tolerance`` (relative) of the random-pair mean,
    i.e. beyond which two frames are no more similar than two random ones.

    Returns (figure, decorrelation_lag). The lag is None if never reached.
    """
    curve = similarity_vs_lag(thumbs, max_lag)
    lags = np.arange(1, max_lag + 1)
    rand_mean, rand_std = random_.mean(), random_.std()

    reached = np.nonzero(curve >= rand_mean * (1 - tolerance))[0]
    lag = int(lags[reached[0]]) if reached.size else None

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4.5))

    ax1.plot(lags, curve, marker="o", markersize=3, label="mean difference at ID distance k")
    ax1.axhline(rand_mean, ls="--", color="gray", label="random pairs (mean)")
    ax1.axhspan(rand_mean - rand_std, rand_mean + rand_std, color="gray", alpha=0.15,
                label="random pairs (±1 std)")
    if lag is not None:
        ax1.axvline(lag, color="red", ls=":", label=f"decorrelation lag = {lag}")
    ax1.set_xlabel("ID distance k")
    ax1.set_ylabel("mean abs pixel difference")
    ax1.set_title("Frame similarity vs ID distance")
    ax1.legend(fontsize=8)

    bins = np.linspace(0, max(consecutive.max(), random_.max()), 40)
    ax2.hist(random_, bins=bins, alpha=0.6, label="random pairs")
    ax2.hist(consecutive, bins=bins, alpha=0.6, label="consecutive IDs")
    ax2.set_xlabel("mean abs pixel difference")
    ax2.set_ylabel("pairs")
    ax2.set_title("Distribution of differences")
    ax2.legend()

    ax3.plot(consecutive, lw=0.8)
    ax3.axhline(rand_mean, ls="--", color="gray", label="random pairs (mean)")
    ax3.set_xlabel("frame index i")
    ax3.set_ylabel("difference between frame i and i+1")
    ax3.set_title("Scene changes along the ID order")
    ax3.legend(fontsize=8)

    fig.tight_layout()
    return fig, lag


def class_summary(stats, names=CLASS_NAMES):
    """Per-class table: pixel share, image presence, coverage when present, loss weight."""
    per = stats["per_image"]
    frac = per / per.sum(1, keepdims=True)
    present = per > 0
    cover = [frac[present[:, c], c].mean() if present[:, c].any() else 0
            for c in range(NUM_CLASSES)]
    return pd.DataFrame({
        "name": names,
        "pixel_%": stats["pixel_counts"] / stats["pixel_counts"].sum() * 100,
        "images_%": present.mean(0) * 100,
        "cover_when_present_%": np.array(cover) * 100,
        "weight": class_weights(stats["pixel_counts"]),
    }).round(2)


def plot_class_distribution(stats, palette=None):
    """Horizontal bars (classes 1..8, sorted): pixel share on log scale + image presence."""
    summ = class_summary(stats).iloc[1:]                  # skip background: not evaluated
    summ = summ.sort_values("pixel_%")
    colors = [palette[i] / 255 for i in summ.index] if palette is not None else None

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    a1.barh(summ["name"], summ["pixel_%"], color=colors, edgecolor="black")
    a1.set_xscale("log"); a1.set_title("Share of pixels (%, log scale)")
    a2.barh(summ["name"], summ["images_%"], color=colors, edgecolor="black")
    a2.set_xlim(0, 100); a2.set_title("Images containing the class (%)")
    for ax, col in ((a1, "pixel_%"), (a2, "images_%")):
        for y, v in enumerate(summ[col]):
            ax.text(v, y, f" {v:.1f}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_cooccurrence(stats):
    """P(column class present | row class present), classes 1..8."""
    present = (stats["per_image"] > 0).astype(int)
    cooc = present.T @ present
    cond = (cooc / np.maximum(np.diag(cooc)[:, None], 1))[1:, 1:]
    names = CLASS_NAMES[1:]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cond, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(8)); ax.set_xticklabels(names, rotation=60, ha="right")
    ax.set_yticks(range(8)); ax.set_yticklabels(names)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{cond[i, j]:.2f}", ha="center", va="center",
                    color="white" if cond[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, label="P(col | row)")
    ax.set_title("Class co-occurrence")
    fig.tight_layout()
    return fig


def spatial_prior(samples, step=16):
    """Mean frequency of each class at every (downsampled) pixel position: (9, h, w)."""
    acc = None
    for s in tqdm(samples, desc="Spatial prior"):
        _, m = load_sample(s)
        m = m[::step, ::step]
        onehot = np.stack([m == c for c in range(NUM_CLASSES)]).astype(np.float32)
        acc = onehot if acc is None else acc + onehot
    return acc / len(samples)


def plot_spatial_prior(prior):
    """One heatmap per class, each with its own colour scale so rare classes are visible."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 5))
    for c, ax in zip(range(1, NUM_CLASSES), axes.ravel()):
        peak = float(prior[c].max())
        im = ax.imshow(prior[c], cmap="magma", vmin=0, vmax=peak)
        ax.set_title(f"{CLASS_NAMES[c]} (max {peak:.3f})", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    return fig


def plot_brightness(stats):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(stats["brightness"], bins=30, edgecolor="black")
    ax.set_xlabel("mean image brightness [0-1]"); ax.set_ylabel("images")
    ax.set_title("Illumination variability")
    return fig


def consecutive_similarity(samples, n_random=200, downscale=16, seed=0):
    """
    param:
        samples: list of sample directories (Path objects)
        n_random: number of random pairs to compare
        downscale: factor to downscale the images for faster comparison
        seed: random seed for reproducibility
        
    Compare consecutive frames with random pairs, to detect leakage risk.

    For each pair we compute the mean absolute pixel difference on small
    grayscale thumbnails. If consecutive pairs are much more similar than
    random pairs, a random train/val split would leak near-duplicates.

    Returns (consecutive, random_, thumbs) where
        consecutive: (N-1,) mean pixel difference between frames i and i+1
        random_: (n_random,) mean pixel difference between random pairs
        thumbs: (N, h, w) grayscale thumbnails of all frames
    """
    def thumb(path):
        with Image.open(path / RGB_NAME) as im:
            return np.asarray(im.convert("L").reduce(downscale), dtype=np.float32)

    thumbs = [thumb(s) for s in tqdm(samples, desc="Thumbnails")]
    consecutive = np.array([np.abs(thumbs[i] - thumbs[i + 1]).mean()
                            for i in range(len(thumbs) - 1)])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(thumbs), size=(n_random, 2))
    random_ = np.array([np.abs(thumbs[a] - thumbs[b]).mean() for a, b in idx if a != b])
    return consecutive, random_, thumbs


def similarity_vs_lag(thumbs, max_lag=60):
    """Mean pixel difference between frames i and i+lag, for lag = 1..max_lag."""
    return np.array([np.mean([np.abs(thumbs[i] - thumbs[i + k]).mean()
                              for i in range(len(thumbs) - k)])
                     for k in range(1, max_lag + 1)])

# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_samples(samples: list[Path], indices: list[int], palette: np.ndarray | None = None):
    """
    Show image / colourised mask side by side for the given sample indices.
    samples: list of sample directories (Path objects)
    indices: list of indices into samples to show 
    palette: optional (256, 3) colour palette to use for the masks. If None, the palette is read from the first sample.
    """
    palette = get_palette(samples[0]) if palette is None else palette
    fig, axes = plt.subplots(len(indices), 2, figsize=(10, 2.7 * len(indices)), squeeze=False)
    for row, i in enumerate(indices):
        image, mask = load_sample(samples[i])
        axes[row, 0].imshow(image)
        axes[row, 1].imshow(colorize(mask, palette))
        axes[row, 0].set_ylabel(samples[i].name)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_title("RGB")
    axes[0, 1].set_title("Label")
    fig.tight_layout()
    return fig


def class_weights(pixel_counts: np.ndarray, ignore_background: bool = True) -> np.ndarray:
    """Inverse-frequency weights, normalised so that they average to 1.

    Candidate weights for ``CrossEntropyLoss(weight=...)``. Background (class 0)
    is not evaluated by the metric, so its weight is set to 0 by default.
    """
    freq = pixel_counts / pixel_counts.sum()
    w = 1.0 / np.maximum(freq, 1e-8)
    if ignore_background:
        w[0] = 0.0
    w = w / w[w > 0].mean()
    return w
