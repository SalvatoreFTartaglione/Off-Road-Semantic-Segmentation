"""Tests of src/split.py.

The split has too many outputs to check by hand, so most tests verify properties that
must hold for ANY input (no overlap between train and validation, a safety gap between
them, whole blocks, reproducibility), plus a few small examples with known answers.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.split import (assign_blocks, block_statistics, build_blocks, load_split, make_split,
                       purge_neighbours, save_split)

N_FRAMES = 300
MIN_BLOCK = 30
MAX_BLOCK = 120


def synthetic_dataset(seed: int = 0):
    """300 frames with class counts per image and frame-to-frame differences.

    Frames 200-299 form a stretch of near-duplicates (very small differences between
    consecutive frames), like the end of the real dataset.
    """
    rng = np.random.default_rng(seed)
    presence_probability = np.array([1.0, 0.7, 0.7, 0.6, 0.12, 0.2, 0.5, 0.95, 0.9])
    present = rng.random((N_FRAMES, 9)) < presence_probability
    per_image = (present * rng.integers(1000, 50000, size=(N_FRAMES, 9))).astype(np.int64)

    consecutive = rng.uniform(25, 60, size=N_FRAMES - 1)
    consecutive[199:] = rng.uniform(5, 10, size=N_FRAMES - 200)        # near-duplicate stretch
    names = [f"{i:04d}" for i in range(N_FRAMES)]
    return names, per_image, consecutive


@pytest.fixture(scope="module")
def dataset():
    return synthetic_dataset()


@pytest.fixture(scope="module")
def split(dataset):
    names, per_image, consecutive = dataset
    return make_split(names, per_image, consecutive, seed=0)


# --- build_blocks ----------------------------------------------------------------------------
def test_blocks_are_contiguous_and_cover_every_frame(dataset):
    _, _, consecutive = dataset
    block_ids = build_blocks(consecutive, MIN_BLOCK, MAX_BLOCK)
    assert len(block_ids) == N_FRAMES
    assert block_ids[0] == 0
    assert set(np.diff(block_ids)) <= {0, 1}                           # ids never skip or go back


def test_blocks_are_cut_only_at_scene_changes_or_when_too_long(dataset):
    _, _, consecutive = dataset
    block_ids = build_blocks(consecutive, MIN_BLOCK, MAX_BLOCK, cut_quantile=0.5)
    threshold = np.quantile(consecutive, 0.5)
    sizes = np.bincount(block_ids)

    for i in np.flatnonzero(np.diff(block_ids)) + 1:                   # first frame of every new block
        previous_size = sizes[block_ids[i - 1]]
        assert previous_size >= MIN_BLOCK
        assert consecutive[i - 1] >= threshold or previous_size >= MAX_BLOCK


def test_block_sizes_respect_the_limits(dataset):
    _, _, consecutive = dataset
    sizes = np.bincount(build_blocks(consecutive, MIN_BLOCK, MAX_BLOCK))
    assert (sizes[:-1] >= MIN_BLOCK).all()                             # the last one may be shorter
    assert (sizes <= MAX_BLOCK + MIN_BLOCK // 2).all()                 # a tiny tail is merged into its neighbour


def test_near_duplicate_stretch_is_not_cut_at_the_minimum_size():
    """Without scene changes a block keeps growing, up to the maximum size.

    120 large differences (every frame is a scene change, the median is 50) followed by
    100 tiny ones (a near-duplicate stretch, below the median).
    """
    consecutive = np.concatenate([np.full(120, 50.0), np.full(100, 5.0)])
    sizes = np.bincount(build_blocks(consecutive, MIN_BLOCK, MAX_BLOCK, cut_quantile=0.5))
    assert sizes.tolist() == [30, 30, 30, 30, 101]                     # the stretch stays in one block


# --- block_statistics ------------------------------------------------------------------------
def test_block_statistics_add_up_to_the_dataset_totals(dataset):
    _, per_image, consecutive = dataset
    block_ids = build_blocks(consecutive, MIN_BLOCK, MAX_BLOCK)
    pixels, images, frames = block_statistics(per_image, block_ids)

    np.testing.assert_array_equal(pixels.sum(0), per_image.sum(0))
    np.testing.assert_array_equal(images.sum(0), (per_image > 0).sum(0))
    assert frames.sum() == N_FRAMES
    np.testing.assert_array_equal(frames, np.bincount(block_ids))


# --- assign_blocks ---------------------------------------------------------------------------
def test_assignment_returns_one_flag_per_block_and_uses_both_sets(dataset):
    _, per_image, consecutive = dataset
    block_ids = build_blocks(consecutive, MIN_BLOCK, MAX_BLOCK)
    pixels, images, frames = block_statistics(per_image, block_ids)
    flags = assign_blocks(pixels, images, frames, val_fraction=0.2, n_trials=50, seed=0)

    assert flags.dtype == bool
    assert flags.shape == frames.shape
    assert flags.any() and not flags.all()


def test_assignment_is_reproducible_with_the_same_seed(dataset):
    _, per_image, consecutive = dataset
    block_ids = build_blocks(consecutive, MIN_BLOCK, MAX_BLOCK)
    stats = block_statistics(per_image, block_ids)
    first = assign_blocks(*stats, val_fraction=0.2, n_trials=30, seed=3)
    second = assign_blocks(*stats, val_fraction=0.2, n_trials=30, seed=3)
    np.testing.assert_array_equal(first, second)


# --- purge_neighbours ------------------------------------------------------------------------
def test_purge_flags_the_training_frames_next_to_validation():
    is_val = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    purged = purge_neighbours(is_val, gap=2)
    assert np.flatnonzero(purged).tolist() == [2, 3, 6, 7]             # two frames on each side
    assert not (purged & is_val).any()                                  # validation frames are never purged


def test_purge_with_zero_gap_flags_nothing():
    is_val = np.array([0, 1, 1, 0], dtype=bool)
    assert not purge_neighbours(is_val, gap=0).any()


# --- make_split ------------------------------------------------------------------------------
def test_every_frame_belongs_to_exactly_one_group(split):
    train, val, purged = split.train_indices, split.val_indices, split.purged_indices
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(purged)) == 0
    assert len(set(val) & set(purged)) == 0
    assert len(train) + len(val) + len(purged) == N_FRAMES


def test_no_training_frame_is_close_to_a_validation_frame(split):
    """The core guarantee against leakage: a gap of purge_gap frames between the sets."""
    distance = np.abs(split.train_indices[:, None] - split.val_indices[None, :]).min()
    assert distance > split.params["purge_gap"]


def test_validation_is_made_of_whole_blocks(split):
    for block in np.unique(split.block_ids):
        in_block = split.block_ids == block
        assert split.is_val[in_block].all() or not split.is_val[in_block].any()


def test_validation_size_is_close_to_the_requested_fraction(split):
    assert 0.12 <= len(split.val_indices) / N_FRAMES <= 0.30


def test_every_class_appears_in_the_validation_images(split, dataset):
    _, per_image, _ = dataset
    assert ((per_image[split.val_indices] > 0).sum(0)[1:] > 0).all()


def test_the_split_is_reproducible(dataset):
    names, per_image, consecutive = dataset
    first = make_split(names, per_image, consecutive, seed=0)
    second = make_split(names, per_image, consecutive, seed=0)
    np.testing.assert_array_equal(first.is_val, second.is_val)
    np.testing.assert_array_equal(first.is_purged, second.is_purged)


def test_inconsistent_lengths_are_rejected(dataset):
    names, per_image, consecutive = dataset
    with pytest.raises(ValueError):
        make_split(names[:-1], per_image, consecutive)
    with pytest.raises(ValueError):
        make_split(names, per_image, consecutive[:-1])


# --- save_split / load_split -----------------------------------------------------------------
def test_saved_split_can_be_loaded_back(split, dataset, tmp_path):
    names, _, _ = dataset
    path = tmp_path / "splits" / "split.json"                          # the folder is created on demand
    save_split(split, path)

    train_samples, val_samples = load_split(path, [Path(n) for n in names])
    assert [p.name for p in train_samples] == [names[i] for i in split.train_indices]
    assert [p.name for p in val_samples] == [names[i] for i in split.val_indices]


def test_saved_file_documents_the_protocol(split, tmp_path):
    path = tmp_path / "split.json"
    save_split(split, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["params"]["seed"] == split.params["seed"]
    assert payload["n_train"] == len(payload["train"]) == len(split.train_indices)
    assert payload["n_val"] == len(payload["val"]) == len(split.val_indices)
    assert payload["n_purged"] == len(payload["purged"]) == len(split.purged_indices)


def test_loading_fails_if_the_dataset_does_not_match_the_split(split, dataset, tmp_path):
    names, _, _ = dataset
    path = tmp_path / "split.json"
    save_split(split, path)
    incomplete = [Path(n) for n in names if n != names[split.train_indices[0]]]
    with pytest.raises(ValueError):
        load_split(path, incomplete)


def test_loading_refuses_samples_with_repeated_names(split, dataset, tmp_path):
    """data/train/0000 and data/test/0000 have the same name: a dictionary of names would silently
    keep one of them, and the split could be applied to the wrong frames."""
    names, _, _ = dataset
    path = tmp_path / "split.json"
    save_split(split, path)
    mixed = [Path("train") / n for n in names] + [Path("test") / names[0]]
    with pytest.raises(ValueError, match="not unique"):
        load_split(path, mixed)
