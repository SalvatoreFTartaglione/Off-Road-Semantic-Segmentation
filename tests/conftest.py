"""Shared fixtures for the test suite.

The tests never touch the real dataset: they work on tiny synthetic samples written
to a temporary directory with the same layout as the real data
(<id>/rgb.jpg and <id>/labels.png in palette mode), so anybody can run them right
after cloning the repository.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

HEIGHT, WIDTH = 64, 96
N_SAMPLES = 12


def _palette() -> list[int]:
    rng = np.random.default_rng(123)
    return rng.integers(0, 256, size=(256, 3), dtype=np.uint8).ravel().tolist()


def _make_mask(index: int) -> np.ndarray:
    """Deterministic mask: sky on top, vegetation, two trail classes, hood at the bottom."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[:16] = 8                                 # sky
    mask[16:32] = 7                               # high vegetation
    mask[32:58, : WIDTH // 2] = (index % 3) + 1   # class 1..3 depending on the sample
    mask[32:58, WIDTH // 2:] = (index % 4) + 4    # class 4..7 depending on the sample
    return mask                                   # rows 58.. stay 0 (background)


def _make_image(mask: np.ndarray, seed: int) -> np.ndarray:
    """Image whose colour depends on the class of every pixel, plus a little noise."""
    rng = np.random.default_rng(seed)
    colors = np.random.default_rng(7).integers(30, 226, size=(9, 3))
    noise = rng.integers(-8, 9, size=(*mask.shape, 3))
    return np.clip(colors[mask] + noise, 0, 255).astype(np.uint8)


def _write_sample(directory: Path, image: np.ndarray, mask: np.ndarray) -> None:
    directory.mkdir(parents=True)
    Image.fromarray(image).save(directory / "rgb.jpg", quality=100, subsampling=0)
    label = Image.new("P", (mask.shape[1], mask.shape[0]))
    label.putdata(mask.ravel().tolist())
    label.putpalette(_palette())
    label.save(directory / "labels.png")


@pytest.fixture(scope="session")
def synthetic_masks() -> list[np.ndarray]:
    """The ground-truth masks of ``synthetic_samples``, in the same order."""
    return [_make_mask(i) for i in range(N_SAMPLES)]


@pytest.fixture(scope="session")
def synthetic_samples(tmp_path_factory, synthetic_masks) -> list[Path]:
    """Sample directories (0000, 0001, ...) with a synthetic rgb.jpg and labels.png."""
    root = tmp_path_factory.mktemp("dataset")
    samples = []
    for i, mask in enumerate(synthetic_masks):
        directory = root / f"{i:04d}"
        _write_sample(directory, _make_image(mask, seed=i), mask)
        samples.append(directory)
    return samples


@pytest.fixture(scope="session")
def half_black_half_white_sample(tmp_path_factory) -> Path:
    """Left half black with label 1, right half white with label 2.

    Its statistics are known exactly (mean 0.5, std 0.5), and brightness tells which
    label a pixel has, which makes it ideal to check image/mask alignment.
    """
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[:, : WIDTH // 2] = 1
    mask[:, WIDTH // 2:] = 2
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:, WIDTH // 2:] = 255
    directory = tmp_path_factory.mktemp("bw") / "0000"
    _write_sample(directory, image, mask)
    return directory
