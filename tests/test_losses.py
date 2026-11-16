"""Tests of src/losses.py.

Each part of the loss is compared with an independent reference: PyTorch for the
cross-entropy, a naive one-hot implementation for the Dice, and hand-computed values
for the class weights.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.data import load_sample
from src.losses import (SegmentationLoss, compute_class_weights, count_class_pixels, create_loss,
                        dice_loss)

# Pixel counts of the classes 0..8. Class 7 is the most frequent semantic class, class 4 the rarest
COUNTS = np.array([100, 5000, 3000, 2000, 10, 50, 400, 8000, 1000])


@pytest.fixture
def logits_and_target():
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 9, 20, 30, generator=generator)
    target = torch.randint(0, 9, (4, 20, 30), generator=generator)
    return logits, target


def one_hot_dice(logits, target, smooth, ignore_background):
    """Naive reference: expands the target to a one-hot tensor."""
    probs = logits.softmax(1)
    onehot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
    if ignore_background:
        valid = (target != 0).unsqueeze(1).float()
        probs, onehot = probs * valid, onehot * valid
    intersection = (probs * onehot).sum((0, 2, 3))
    cardinality = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    dice = (2 * intersection + smooth) / (cardinality + smooth)
    return 1 - dice[1:].mean()


def perfect_logits(target):
    return F.one_hot(target, 9).permute(0, 3, 1, 2).float() * 50


# --- Class weights -------------------------------------------------------------------------
def test_weights_have_the_expected_shape_and_type():
    weights = compute_class_weights(COUNTS)
    assert weights.shape == (9,)
    assert weights.dtype == torch.float32


def test_most_frequent_semantic_class_has_weight_one_and_rarer_classes_more():
    weights = compute_class_weights(COUNTS, max_weight=1e9)
    assert weights[7] == pytest.approx(1.0)
    assert (weights[1:] >= 1.0).all()
    assert weights[1:].argmax() == 3                      # class 4 is the rarest (index 3 of 1..8)


def test_weight_formula_with_square_root():
    weights = compute_class_weights(COUNTS, power=0.5, max_weight=1e9)
    assert weights[4] == pytest.approx((8000 / 10) ** 0.5, rel=1e-5)
    assert weights[1] == pytest.approx((8000 / 5000) ** 0.5, rel=1e-5)


def test_power_zero_gives_uniform_weights():
    assert torch.allclose(compute_class_weights(COUNTS, power=0.0)[1:], torch.ones(8))


def test_weights_are_capped():
    weights = compute_class_weights(COUNTS, power=1.0, max_weight=20.0)
    assert weights.max() == pytest.approx(20.0)           # 8000 / 10 = 800 would be the raw weight


def test_missing_class_gets_the_maximum_weight_instead_of_breaking():
    counts = COUNTS.copy()
    counts[4] = 0
    weights = compute_class_weights(counts, max_weight=20.0)
    assert torch.isfinite(weights).all()
    assert weights[4] == pytest.approx(20.0)


def test_background_weight_is_the_configured_constant():
    assert compute_class_weights(COUNTS, background_weight=2.5)[0] == pytest.approx(2.5)


def test_count_class_pixels_matches_manual_count(synthetic_samples):
    counts = count_class_pixels(synthetic_samples, image_size=None)
    expected = np.zeros(9, dtype=np.int64)
    for sample in synthetic_samples:
        expected += np.bincount(load_sample(sample)[1].ravel(), minlength=9)
    np.testing.assert_array_equal(counts, expected)


def test_count_class_pixels_follows_the_working_resolution(synthetic_samples):
    counts = count_class_pixels(synthetic_samples, image_size=(32, 48))
    assert counts.sum() == len(synthetic_samples) * 32 * 48


# --- Dice ------------------------------------------------------------------------------------
@pytest.mark.parametrize("ignore_background", [False, True])
def test_dice_matches_the_naive_one_hot_implementation(logits_and_target, ignore_background):
    logits, target = logits_and_target
    ours = dice_loss(logits, target, smooth=1.0, ignore_background=ignore_background)
    reference = one_hot_dice(logits, target, 1.0, ignore_background)
    assert torch.allclose(ours, reference, atol=1e-6)


def test_dice_is_zero_for_a_perfect_prediction_and_large_for_a_wrong_one(logits_and_target):
    _, target = logits_and_target
    perfect = dice_loss(perfect_logits(target), target)
    wrong = dice_loss(perfect_logits((target + 1) % 9), target)
    assert perfect.item() == pytest.approx(0.0, abs=1e-3)
    assert wrong.item() > 0.9


def test_dice_stays_between_zero_and_one(logits_and_target):
    logits, target = logits_and_target
    assert 0.0 <= dice_loss(logits, target).item() <= 1.0


# --- Combined loss ---------------------------------------------------------------------------
def test_cross_entropy_term_matches_pytorch(logits_and_target):
    logits, target = logits_and_target
    weights = compute_class_weights(COUNTS)
    ours = SegmentationLoss(weights).components(logits, target)["ce"]
    assert torch.allclose(ours, F.cross_entropy(logits, target, weight=weights))


def test_ignore_background_uses_ignore_index_in_the_cross_entropy(logits_and_target):
    logits, target = logits_and_target
    weights = compute_class_weights(COUNTS)
    ours = SegmentationLoss(weights, ignore_background=True).components(logits, target)["ce"]
    assert torch.allclose(ours, F.cross_entropy(logits, target, weight=weights, ignore_index=0))


def test_perfect_prediction_gives_zero_loss(logits_and_target):
    _, target = logits_and_target
    parts = SegmentationLoss(compute_class_weights(COUNTS)).components(perfect_logits(target), target)
    assert parts["ce"].item() == pytest.approx(0.0, abs=1e-3)
    assert parts["dice"].item() == pytest.approx(0.0, abs=1e-3)
    assert parts["total"].item() == pytest.approx(0.0, abs=2e-3)


def test_background_pixels_are_really_ignored_when_requested(logits_and_target):
    logits, target = logits_and_target
    changed = logits.clone()
    background = (target == 0).unsqueeze(1).expand_as(changed)
    changed[background] = torch.randn(int(background.sum()))

    weights = compute_class_weights(COUNTS)
    ignoring = SegmentationLoss(weights, ignore_background=True)
    keeping = SegmentationLoss(weights, ignore_background=False)

    assert torch.allclose(ignoring(logits, target), ignoring(changed, target), atol=1e-6)
    assert not torch.allclose(keeping(logits, target), keeping(changed, target), atol=1e-6)


def test_total_is_the_weighted_sum_of_the_two_terms(logits_and_target):
    logits, target = logits_and_target
    weights = compute_class_weights(COUNTS)
    base = SegmentationLoss(weights).components(logits, target)
    scaled = SegmentationLoss(weights, ce_weight=2.0, dice_weight=0.5).components(logits, target)
    assert scaled["total"].item() == pytest.approx(2.0 * base["ce"].item() + 0.5 * base["dice"].item(),
                                                   rel=1e-5)


def test_a_term_with_zero_weight_is_skipped(logits_and_target):
    logits, target = logits_and_target
    parts = SegmentationLoss(compute_class_weights(COUNTS), dice_weight=0.0).components(logits, target)
    assert parts["dice"].item() == 0.0
    assert parts["total"].item() == pytest.approx(parts["ce"].item())


def test_wrong_shapes_are_rejected(logits_and_target):
    logits, target = logits_and_target
    loss = SegmentationLoss(compute_class_weights(COUNTS))
    with pytest.raises(ValueError):
        loss(logits, target.unsqueeze(1))
    with pytest.raises(ValueError):
        loss(logits[0], target)


def test_class_weights_are_a_buffer_that_follows_the_module():
    loss = SegmentationLoss(compute_class_weights(COUNTS))
    assert "class_weights" in dict(loss.named_buffers())
    assert "class_weights" in loss.state_dict()
    assert not any(p.requires_grad for p in loss.parameters())


def test_gradients_are_finite(logits_and_target):
    logits, target = logits_and_target
    logits = logits.clone().requires_grad_(True)
    SegmentationLoss(compute_class_weights(COUNTS))(logits, target).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_the_loss_can_be_minimised_by_a_small_network():
    torch.manual_seed(0)
    images = torch.randn(8, 3, 16, 16)
    target = (images[:, 0] > 0).long() + 2 * (images[:, 1] > 0).long()    # 4 classes, linearly separable
    model = nn.Conv2d(3, 9, kernel_size=1)
    loss_fn = SegmentationLoss(compute_class_weights(COUNTS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    first = loss_fn(model(images), target).item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = loss_fn(model(images), target)
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.5 * first


def test_create_loss_computes_weights_from_the_given_samples(synthetic_samples):
    loss = create_loss(synthetic_samples, image_size=None)
    expected = compute_class_weights(count_class_pixels(synthetic_samples, image_size=None))
    assert torch.allclose(loss.class_weights, expected)
