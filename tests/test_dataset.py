"""Tests of src/data.py (loading) and src/dataset.py (augmentation, dataset, dataloaders).

The most important test is the alignment one: whatever geometric augmentation is
applied, a pixel must keep the label that belongs to it.
"""
import numpy as np
import pytest
import torch

from src import config
from src.data import get_palette, load_sample
from src.dataset import (SegmentationDataset, build_transforms, compute_normalization, create_dataloaders, denormalize, resize_pair)

SIZE = (32, 48)                                   # working resolution used in the tests (height, width)
MEAN, STD = (0.4, 0.45, 0.42), (0.26, 0.26, 0.27)


# --- Loading ---------------------------------------------------------------------------------
def test_load_sample_returns_the_class_indices_and_not_the_palette_colours(synthetic_samples, synthetic_masks):
    image, mask = load_sample(synthetic_samples[3])
    assert image.shape == (*synthetic_masks[3].shape, 3) and image.dtype == np.uint8
    assert mask.ndim == 2 and mask.dtype == np.uint8
    np.testing.assert_array_equal(mask, synthetic_masks[3])


def test_palette_has_256_rgb_entries(synthetic_samples):
    assert get_palette(synthetic_samples[0]).shape == (256, 3)


# --- Resizing and statistics -----------------------------------------------------------------
def test_resize_returns_the_requested_shapes(synthetic_samples):
    image, mask = load_sample(synthetic_samples[5])
    small_image, small_mask = resize_pair(image, mask, SIZE)
    assert small_image.shape == (*SIZE, 3) and small_image.dtype == np.uint8
    assert small_mask.shape == SIZE and small_mask.dtype == np.uint8


def test_resize_does_not_invent_labels():
    """A noisy mask with only labels 1 and 8, resized by a non-integer factor.

    An interpolating resize would produce in-between values (the average of 1 and 8 is
    neither of them), while nearest neighbour can only return labels that already exist.
    """
    rng = np.random.default_rng(0)
    mask = rng.choice([1, 8], size=(64, 96)).astype(np.uint8)
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    _, small_mask = resize_pair(image, mask, (25, 37))
    assert set(np.unique(small_mask)) <= {1, 8}


@pytest.mark.parametrize("image_size", [None, SIZE])
def test_normalisation_statistics_of_a_known_image(half_black_half_white_sample, image_size):
    mean, std = compute_normalization([half_black_half_white_sample], image_size)
    np.testing.assert_allclose(mean, [0.5] * 3, atol=0.02)
    np.testing.assert_allclose(std, [0.5] * 3, atol=0.02)


# --- Transforms ------------------------------------------------------------------------------
def test_validation_transform_is_deterministic_and_reversible(synthetic_samples):
    image, mask = resize_pair(*load_sample(synthetic_samples[0]), SIZE)
    transform = build_transforms(MEAN, STD, train=False)
    first, second = transform(image=image, mask=mask), transform(image=image, mask=mask)

    assert torch.equal(first["image"], second["image"])
    assert torch.equal(first["mask"], second["mask"])
    assert first["image"].shape == (3, *SIZE) and first["image"].dtype == torch.float32
    restored = denormalize(first["image"], MEAN, STD)
    assert np.abs(restored.astype(int) - image.astype(int)).max() <= 1  # only rounding differences


def test_training_transform_is_reproducible_with_the_same_seed(synthetic_samples):
    image, mask = resize_pair(*load_sample(synthetic_samples[1]), SIZE)
    first = build_transforms(MEAN, STD, train=True, seed=5)
    second = build_transforms(MEAN, STD, train=True, seed=5)
    for _ in range(5):
        a, b = first(image=image, mask=mask), second(image=image, mask=mask)
        assert torch.equal(a["image"], b["image"])
        assert torch.equal(a["mask"], b["mask"])


def test_training_transform_actually_changes_the_image(synthetic_samples):
    image, mask = resize_pair(*load_sample(synthetic_samples[1]), SIZE)
    reference = build_transforms(MEAN, STD, train=False)(image=image, mask=mask)["image"]
    outputs = [build_transforms(MEAN, STD, train=True, seed=s)(image=image, mask=mask)["image"]
               for s in range(10)]
    assert any(not torch.equal(o, reference) for o in outputs)


PROFILES = sorted(config.AUGMENTATIONS)


@pytest.mark.parametrize("profile", PROFILES)
def test_augmentation_never_creates_labels_that_were_not_there(synthetic_samples, profile):
    image, mask = resize_pair(*load_sample(synthetic_samples[2]), SIZE)
    allowed = set(np.unique(mask)) | {0}                               # padding is labelled 0
    for seed in range(40):
        out = build_transforms(MEAN, STD, train=True, seed=seed, augmentation=profile)(image=image, mask=mask)
        assert out["mask"].dtype == torch.uint8
        assert set(np.unique(out["mask"].numpy())) <= allowed


@pytest.mark.parametrize("profile", PROFILES)
def test_image_and_mask_stay_aligned_after_augmentation(half_black_half_white_sample, profile):
    """Black pixels are labelled 1 and white pixels 2: this must remain true after flips,
    zooms, shifts and rotations, otherwise the network would learn from wrong labels."""
    image, mask = resize_pair(*load_sample(half_black_half_white_sample), SIZE)
    for seed in range(30):
        out = build_transforms(MEAN, STD, train=True, seed=seed, augmentation=profile)(image=image, mask=mask)
        brightness = denormalize(out["image"], MEAN, STD).mean(axis=2)
        labels = out["mask"].numpy()
        dark, bright = brightness[labels == 1], brightness[labels == 2]
        assert dark.size > 100 and bright.size > 100
        assert bright.mean() > dark.mean() + 30        # a misaligned mask would give a negative difference


@pytest.mark.parametrize("profile", PROFILES)
def test_every_profile_is_reproducible_with_the_same_seed(synthetic_samples, profile):
    image, mask = resize_pair(*load_sample(synthetic_samples[1]), SIZE)
    first = build_transforms(MEAN, STD, train=True, seed=3, augmentation=profile)
    second = build_transforms(MEAN, STD, train=True, seed=3, augmentation=profile)
    for _ in range(5):
        a, b = first(image=image, mask=mask), second(image=image, mask=mask)
        assert torch.equal(a["image"], b["image"]) and torch.equal(a["mask"], b["mask"])


def test_standard_profile_is_the_pipeline_used_by_experiments_a_b_and_c(synthetic_samples):
    """Regression test: the values below are the ones that were hard-coded before the profiles
    existed. If the "standard" profile changes, A, B, C and E are no longer reproducible."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    historical = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.85, 1.25), rotate=(-5, 5), translate_percent=(-0.05, 0.05), fill=0, fill_mask=0, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.RandomGamma(gamma_limit=(75, 130), p=0.3),
        A.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.2, hue=0.0, p=0.3),
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0), ToTensorV2(),
    ], seed=11)
    current = build_transforms(MEAN, STD, train=True, seed=11, augmentation="standard")
    image, mask = resize_pair(*load_sample(synthetic_samples[4]), SIZE)
    for _ in range(8):
        a, b = historical(image=image, mask=mask), current(image=image, mask=mask)
        assert torch.equal(a["image"], b["image"]) and torch.equal(a["mask"], b["mask"])


def test_strong_profile_adds_transforms_and_standard_does_not(synthetic_samples):
    names = lambda profile: [type(t).__name__ for t in build_transforms(MEAN, STD, True, 0, profile).transforms]
    assert {"GaussianBlur", "GaussNoise", "CoarseDropout"} <= set(names("strong"))
    assert not {"GaussianBlur", "GaussNoise", "CoarseDropout"} & set(names("standard"))


def test_strong_augmentation_changes_the_images_more_than_the_standard_one(synthetic_samples):
    image, mask = resize_pair(*load_sample(synthetic_samples[1]), SIZE)
    reference = build_transforms(MEAN, STD, train=False)(image=image, mask=mask)["image"]

    def mean_change(profile):
        outs = [build_transforms(MEAN, STD, True, s, profile)(image=image, mask=mask)["image"] for s in range(60)]
        return float(np.mean([(o - reference).abs().mean().item() for o in outs]))

    assert mean_change("strong") > mean_change("standard")


def test_coarse_dropout_darkens_the_image_but_never_touches_the_mask(synthetic_samples, monkeypatch):
    """Only dropout is enabled (always applied): the mask must be identical to the input one and some
    pixels of the image must be black."""
    only_dropout = {**config.AUGMENTATIONS["strong"], "flip_p": 0.0, "affine_p": 0.0,
                    "brightness_contrast_p": 0.0, "gamma_p": 0.0, "saturation_p": 0.0, "blur_p": 0.0,
                    "noise_p": 0.0, "dropout_p": 1.0}
    monkeypatch.setitem(config.AUGMENTATIONS, "only_dropout", only_dropout)
    image = np.full((*SIZE, 3), 200, dtype=np.uint8)
    mask = np.random.default_rng(0).integers(1, 9, size=SIZE).astype(np.uint8)
    out = build_transforms((0.0,) * 3, (1.0,) * 3, True, 0, "only_dropout")(image=image, mask=mask)
    np.testing.assert_array_equal(out["mask"].numpy(), mask)
    restored = denormalize(out["image"], (0.0,) * 3, (1.0,) * 3)
    assert (restored == 0).any() and (restored == 200).any()


def test_unknown_augmentation_profile_is_rejected():
    with pytest.raises(ValueError):
        build_transforms(MEAN, STD, train=True, augmentation="does_not_exist")


# --- Dataset ---------------------------------------------------------------------------------
def test_dataset_items_have_the_format_expected_by_the_loss(synthetic_samples):
    dataset = SegmentationDataset(synthetic_samples, build_transforms(MEAN, STD, train=False), SIZE)
    image, mask = dataset[0]
    assert len(dataset) == len(synthetic_samples)
    assert image.shape == (3, *SIZE) and image.dtype == torch.float32
    assert mask.shape == SIZE and mask.dtype == torch.int64
    assert 0 <= mask.min() and mask.max() <= 8


def test_cache_does_not_change_the_items(synthetic_samples):
    transform = build_transforms(MEAN, STD, train=False)
    cached = SegmentationDataset(synthetic_samples[:4], transform, SIZE, cache=True)
    streamed = SegmentationDataset(synthetic_samples[:4], transform, SIZE, cache=False)
    for i in range(4):
        assert torch.equal(cached[i][0], streamed[i][0])
        assert torch.equal(cached[i][1], streamed[i][1])


def test_augmentation_does_not_modify_the_cached_samples(synthetic_samples):
    dataset = SegmentationDataset(synthetic_samples[:3], build_transforms(MEAN, STD, train=True, seed=0), SIZE)
    before = [(image.copy(), mask.copy()) for image, mask in dataset._cache]
    for _ in range(5):
        for i in range(3):
            dataset[i]
    for (image, mask), (old_image, old_mask) in zip(dataset._cache, before):
        np.testing.assert_array_equal(image, old_image)
        np.testing.assert_array_equal(mask, old_mask)


# --- Dataloaders -----------------------------------------------------------------------------
def test_dataloader_batches_and_sizes(synthetic_samples):
    train, val = synthetic_samples[:8], synthetic_samples[8:]
    train_loader, val_loader, mean, std = create_dataloaders(train, val, image_size=SIZE, batch_size=3)

    assert len(train_loader) == 2                                      # 8 // 3: the incomplete batch is dropped
    assert len(val_loader) == 2                                        # ceil(4 / 3): validation keeps everything
    images, masks = next(iter(train_loader))
    assert images.shape == (3, 3, *SIZE) and images.dtype == torch.float32
    assert masks.shape == (3, *SIZE) and masks.dtype == torch.int64
    assert len(mean) == len(std) == 3


def test_normalisation_comes_from_the_training_samples_only(synthetic_samples):
    train, val = synthetic_samples[:8], synthetic_samples[8:]
    _, _, mean, std = create_dataloaders(train, val, image_size=SIZE, batch_size=4)
    expected_mean, expected_std = compute_normalization(train, SIZE)
    np.testing.assert_allclose(mean, expected_mean)
    np.testing.assert_allclose(std, expected_std)


def test_validation_batches_are_not_shuffled_nor_augmented(synthetic_samples):
    train, val = synthetic_samples[:8], synthetic_samples[8:]
    _, val_loader, mean, std = create_dataloaders(train, val, image_size=SIZE, batch_size=4)
    first = torch.cat([m for _, m in val_loader])
    second = torch.cat([m for _, m in val_loader])
    assert torch.equal(first, second)
    for i, sample in enumerate(val):
        _, expected = resize_pair(*load_sample(sample), SIZE)
        np.testing.assert_array_equal(first[i].numpy(), expected)


def test_dataloaders_are_reproducible_with_the_same_seed(synthetic_samples):
    train, val = synthetic_samples[:8], synthetic_samples[8:]
    first = next(iter(create_dataloaders(train, val, image_size=SIZE, batch_size=4, seed=7)[0]))
    second = next(iter(create_dataloaders(train, val, image_size=SIZE, batch_size=4, seed=7)[0]))
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
