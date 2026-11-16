"""Tests of src/metrics.py.

The reference is the code of the official evaluation notebook, copied verbatim: the
metric of this project must give exactly the same numbers.
"""
import numpy as np
import pytest
import torch

from src.metrics import MeanIoU, metrics_report, per_image_iou


# --- Verbatim copies of the functions of the official evaluation notebook ---------------
def official_iou(mask1, mask2, label):
    intersection = np.sum((mask1 == label) & (mask2 == label))
    union = np.sum((mask1 == label) | (mask2 == label))
    if union == 0:
        return np.nan
    return intersection / union


def official_all_iou(mask1, mask2, num_labels=8):
    iou_scores = np.zeros((num_labels))
    for label in range(num_labels):
        iou_scores[label] = official_iou(mask1, mask2, label + 1)
    return iou_scores


def official_matrix(prediction: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    return np.stack([official_all_iou(t.numpy(), p.numpy()) for t, p in zip(target, prediction)])


def random_pair(seed: int, batch: int = 6, size=(24, 32)):
    """Ground truth and noisy prediction where every image only contains a few classes.

    Predictions are noisy but only use classes 0-4, so classes 5-8 are often empty in
    both masks (NaN) while classes 1-4 can be predicted where they are absent (IoU 0).
    """
    rng = np.random.default_rng(seed)
    target = np.empty((batch, *size), dtype=np.int64)
    for i in range(batch):
        pool = rng.choice(9, size=rng.integers(2, 5), replace=False)
        target[i] = rng.choice(pool, size=size)
    prediction = target.copy()
    noisy = rng.random(target.shape) < 0.25
    prediction[noisy] = rng.integers(0, 5, size=int(noisy.sum()))
    return torch.from_numpy(prediction), torch.from_numpy(target)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_iou_matrix_matches_official_code(seed):
    prediction, target = random_pair(seed)
    expected = official_matrix(prediction, target)
    assert np.isnan(expected).any(), "the test data should contain NaN entries"
    assert (expected == 0).any(), "the test data should contain zero-IoU entries"
    np.testing.assert_allclose(per_image_iou(prediction, target), expected, equal_nan=True)


def test_summary_matches_official_code():
    prediction, target = random_pair(seed=11)
    expected = official_matrix(prediction, target)

    metric = MeanIoU()
    metric.update(prediction, target)
    results = metric.compute()

    assert results["score"] == pytest.approx(np.nanmean(expected))
    np.testing.assert_allclose(results["per_class"], np.nanmean(expected, axis=0), equal_nan=True)
    np.testing.assert_allclose(results["per_image"], np.nanmean(expected, axis=1), equal_nan=True)
    assert results["mean_class_iou"] == pytest.approx(np.nanmean(np.nanmean(expected, axis=0)))


def test_accumulating_batches_equals_a_single_update():
    prediction, target = random_pair(seed=5, batch=7)
    single = MeanIoU()
    single.update(prediction, target)

    chunked = MeanIoU()
    for start in range(0, 7, 3):
        chunked.update(prediction[start:start + 3], target[start:start + 3])

    np.testing.assert_allclose(chunked.compute()["iou_matrix"], single.compute()["iou_matrix"],
                               equal_nan=True)


def test_logits_and_labels_give_the_same_result():
    prediction, target = random_pair(seed=2)
    logits = torch.nn.functional.one_hot(prediction, 9).permute(0, 3, 1, 2).float() * 10

    from_labels, from_logits = MeanIoU(), MeanIoU()
    from_labels.update(prediction, target)
    from_logits.update(logits, target)

    np.testing.assert_allclose(from_logits.compute()["iou_matrix"],
                               from_labels.compute()["iou_matrix"], equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_tensors_give_the_same_result_as_cpu():
    prediction, target = random_pair(seed=4)
    on_cpu = per_image_iou(prediction, target)
    on_gpu = per_image_iou(prediction.cuda(), target.cuda())
    np.testing.assert_allclose(on_gpu, on_cpu, equal_nan=True)


def test_perfect_prediction_gives_one_for_every_present_class():
    labels = torch.randint(0, 9, (2, 16, 16))
    iou = per_image_iou(labels, labels)
    assert np.nanmean(iou) == 1.0
    assert np.all(iou[~np.isnan(iou)] == 1.0)


def test_class_absent_from_both_masks_is_nan():
    empty = torch.zeros(1, 4, 4, dtype=torch.long)
    assert np.isnan(per_image_iou(empty, empty)).all()


def test_class_predicted_but_absent_from_the_ground_truth_scores_zero():
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    prediction = target.clone()
    prediction[0, 0, 0] = 3
    iou = per_image_iou(prediction, target)[0]
    assert iou[2] == 0.0                                  # class 3 is column index 2
    assert np.isnan(np.delete(iou, 2)).all()


def test_false_positives_on_background_lower_the_iou_of_the_class():
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, 0, :2] = 3                                  # 2 pixels of class 3
    prediction = target.clone()
    prediction[0, 3, :2] = 3                              # 2 more pixels of class 3 on the background
    # intersection 2, union 4
    assert per_image_iou(prediction, target)[0, 2] == pytest.approx(0.5)


def test_reset_clears_the_accumulated_scores():
    prediction, target = random_pair(seed=1)
    metric = MeanIoU()
    metric.update(prediction, target)
    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()


def test_compute_before_any_update_raises():
    with pytest.raises(RuntimeError):
        MeanIoU().compute()


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        per_image_iou(torch.zeros(2, 4, 4, dtype=torch.long), torch.zeros(2, 4, 5, dtype=torch.long))


def test_report_counts_the_images_where_each_class_counted():
    prediction, target = random_pair(seed=7)
    metric = MeanIoU()
    metric.update(prediction, target)
    results = metric.compute()
    report = metrics_report(results)

    assert list(report.index) == list(range(1, 9))
    assert list(report.columns) == ["name", "iou", "images"]
    np.testing.assert_array_equal(report["images"].to_numpy(), (~np.isnan(results["iou_matrix"])).sum(0))
