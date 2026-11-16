"""Tests of src/predict.py: shapes and types of the prediction, independence from the batch size,
post-processing and the metric (compared with MeanIoU and with hand-computed values).
"""
import numpy as np
import pytest
import torch

import src.predict as predict_module
from src import config
from src.metrics import MeanIoU
from src.models import UNet
from src.predict import (Segmenter, compute_all_iou, compute_iou, evaluate, load_model, predict,
                         suppress_small_classes)

SIZE = (32, 48)                                       # working resolution of the tiny test network
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    model = UNet(base_channels=4, depth=2)
    path = tmp_path_factory.mktemp("weights") / "final.pt"
    torch.save({"model_name": "unet", "model_kwargs": {"base_channels": 4, "depth": 2}, "mean": (0.4, 0.45, 0.42),
                "std": (0.26, 0.26, 0.27), "image_size": SIZE, "model": model.state_dict(), "epoch": 1}, path)
    return path


@pytest.fixture()
def segmenter(checkpoint):
    return load_model(checkpoint, device="cpu", min_area=0.0)


def random_images(n, height=60, width=90, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(n, height, width, 3), dtype=np.uint8)


# --- Loading ---------------------------------------------------------------------------------
def test_load_model_restores_the_inference_settings_of_the_checkpoint(segmenter):
    assert isinstance(segmenter, Segmenter) and isinstance(segmenter.model, UNet)
    assert not segmenter.model.training
    assert segmenter.mean == pytest.approx((0.4, 0.45, 0.42)) and segmenter.image_size == SIZE
    assert segmenter.device == CPU


def test_post_processing_settings_come_from_the_config(checkpoint):
    seg = load_model(checkpoint, device="cpu")
    assert seg.min_area == config.POSTPROCESS_MIN_AREA and seg.rare_classes == config.RARE_CLASSES
    assert seg.without_postprocessing().min_area == 0.0 and seg.min_area == config.POSTPROCESS_MIN_AREA


# --- Shapes, types, batch independence -------------------------------------------------------
def test_one_image_gives_a_label_map_of_the_same_size(segmenter):
    out = predict(segmenter, random_images(1, 60, 90)[0])
    assert out.shape == (60, 90) and out.dtype == np.uint8 and out.max() <= 8


@pytest.mark.parametrize("batch", [1, 2, 5, 19])
def test_a_batch_gives_the_shape_required_by_the_project(segmenter, batch):
    """(batch_size, rows, cols, 3) uint8 -> (batch_size, rows, cols, 1) uint8, for any batch size
    (19 is larger than the internal chunk of predict)."""
    segmenter.batch_size = 4
    out = predict(segmenter, random_images(batch, 40, 70))
    assert out.shape == (batch, 40, 70, 1) and out.dtype == np.uint8


def test_the_prediction_does_not_depend_on_the_batch_it_is_in(segmenter):
    images = random_images(6, 50, 50, seed=3)
    together = predict(segmenter, images)
    for i, image in enumerate(images):
        assert np.array_equal(predict(segmenter, image), together[i, ..., 0])


def test_images_of_different_sizes_are_supported_and_the_input_is_not_modified(segmenter):
    image = random_images(1, 123, 77, seed=4)[0]
    before = image.copy()
    assert predict(segmenter, image).shape == (123, 77)
    assert np.array_equal(image, before)


def test_the_input_is_checked(segmenter):
    with pytest.raises(ValueError):
        predict(segmenter, np.zeros((10, 10, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        predict(segmenter, np.zeros((10, 10), dtype=np.uint8))


def test_predict_works_without_the_model_argument(checkpoint, monkeypatch):
    monkeypatch.setattr(config, "FINAL_WEIGHTS", checkpoint)
    monkeypatch.setattr(predict_module, "_DEFAULT_SEGMENTER", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    images = random_images(3, 40, 60)
    out = predict(images)
    assert out.shape == (3, 40, 60, 1)
    assert predict_module._DEFAULT_SEGMENTER is not None        # loaded once, then reused


def test_preprocessing_matches_the_training_resolution_and_normalisation(segmenter):
    image = np.full((64, 96, 3), 128, dtype=np.uint8)
    batch = predict_module._preprocess(image[None], segmenter)
    assert batch.shape == (1, 3, *SIZE)
    expected = (128 / 255 - torch.tensor(segmenter.mean)) / torch.tensor(segmenter.std)
    assert torch.allclose(batch[0, :, 0, 0], expected, atol=1e-5)


# --- Post-processing -------------------------------------------------------------------------
def scores_with_regions(regions, size=(10, 10), classes=9):
    """Scores where each (class, rows, cols) region wins, everything else is class 0."""
    scores = torch.zeros(1, classes, *size)
    scores[:, 0] = 1.0
    for c, rows, cols in regions:
        scores[0, c, rows, cols] = 2.0
    return scores


def test_small_classes_are_removed_and_their_pixels_go_to_the_next_best_class():
    scores = scores_with_regions([(3, slice(0, 5), slice(0, 10)),      # 50 pixels: kept
                                  (7, slice(8, 9), slice(0, 2))])      # 2 pixels: removed
    scores[0, 2, 8, 0:2] = 1.5                                         # the second best class of those pixels
    plain = suppress_small_classes(scores, 0.0)
    cleaned = suppress_small_classes(scores, 0.05)
    assert (plain[0, 8, 0:2] == 7).all()
    assert (cleaned[0, 8, 0:2] == 2).all()                             # went to class 2, not to the background
    assert (cleaned[0, 0:5] == 3).all()                                # the large class is untouched


def test_the_background_and_the_rare_classes_are_not_removed():
    scores = scores_with_regions([(4, slice(0, 1), slice(0, 1)), (5, slice(1, 2), slice(0, 1)),
                                  (6, slice(2, 3), slice(0, 1))])
    cleaned = suppress_small_classes(scores, 0.05, rare_min_area=0.0, rare_classes=(4, 5))
    assert cleaned[0, 0, 0] == 4 and cleaned[0, 1, 0] == 5             # puddle and obstacle survive
    assert cleaned[0, 2, 0] != 6                                       # an ordinary class does not
    assert (suppress_small_classes(scores, 0.05)[0] == 0).sum() > 0    # the background is still there


def test_the_threshold_is_a_fraction_of_the_image_so_it_does_not_depend_on_the_resolution():
    small = scores_with_regions([(3, slice(0, 1), slice(0, 4))], size=(10, 10))                       # 4%
    large = scores_with_regions([(3, slice(0, 2), slice(0, 8))], size=(20, 20))                       # 4%
    for scores in (small, large):
        assert (suppress_small_classes(scores, 0.03) == 3).any()
        assert not (suppress_small_classes(scores, 0.05) == 3).any()


def test_post_processing_is_independent_for_every_image_of_the_batch():
    scores = torch.cat([scores_with_regions([(3, slice(0, 1), slice(0, 1))]),
                        scores_with_regions([(3, slice(0, 6), slice(0, 10))])])
    cleaned = suppress_small_classes(scores, 0.05)
    assert not (cleaned[0] == 3).any() and (cleaned[1] == 3).any()


# --- The metric ------------------------------------------------------------------------------
def test_iou_of_known_masks():
    a = np.array([[1, 1, 2], [2, 0, 0]])
    b = np.array([[1, 2, 2], [2, 2, 0]])
    assert compute_iou(a, b, 1) == pytest.approx(1 / 2)                # {(0,0)} / {(0,0),(0,1)}
    assert compute_iou(a, b, 2) == pytest.approx(2 / 4)
    assert np.isnan(compute_iou(a, b, 5))                              # in neither map


def test_a_class_predicted_but_absent_from_the_ground_truth_scores_zero_and_one_in_neither_is_ignored():
    truth = np.zeros((4, 4), dtype=np.uint8)
    prediction = truth.copy()
    prediction[0, 0] = 3
    ious = compute_all_iou(truth, prediction)
    assert ious.shape == (8,) and ious[2] == 0.0 and np.isnan(np.delete(ious, 2)).all()


def test_the_metric_agrees_with_the_one_used_during_training():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 9, size=(6, 20, 30))
    prediction = np.where(rng.random(truth.shape) < 0.6, truth, rng.integers(0, 9, size=truth.shape))
    ours = np.stack([compute_all_iou(t, p) for t, p in zip(truth, prediction)])
    metric = MeanIoU()
    metric.update(torch.from_numpy(np.eye(9)[prediction].transpose(0, 3, 1, 2)), torch.from_numpy(truth))
    assert np.nanmean(ours) == pytest.approx(metric.compute()["score"])


# --- evaluate --------------------------------------------------------------------------------
def test_evaluate_scores_a_folder_of_samples(segmenter, synthetic_samples):
    result = evaluate(segmenter, synthetic_samples[0].parent, limit=5, verbose=False)
    assert result.iou.shape == (5, 8) and len(result.samples) == 5
    assert result.score == pytest.approx(np.nanmean(result.iou))
    assert result.per_class.shape == (8,) and result.per_image.shape == (5,)
    assert result.fps > 0 and result.peak_gpu_gb == 0.0
    assert list(result.table().columns) == ["mean IoU", "valid images", "IoU = 0"]


def test_a_perfect_network_scores_one(synthetic_samples, monkeypatch):
    """predict is replaced by the ground truth itself: the evaluation loop must then give exactly 1."""
    from src.data import load_sample
    truth = {}

    def oracle(model, image):
        return truth[image.tobytes()]

    for sample in synthetic_samples[:4]:
        image, labels = load_sample(sample)
        truth[image.tobytes()] = labels
    monkeypatch.setattr(predict_module, "predict", oracle)
    # evaluate needs a Segmenter only for the device: use a minimal one
    seg = Segmenter(model=UNet(base_channels=2, depth=1), mean=(0, 0, 0), std=(1, 1, 1), image_size=SIZE, device=CPU)
    result = evaluate(seg, synthetic_samples[0].parent, limit=4, verbose=False)
    assert result.score == pytest.approx(1.0)


def test_evaluate_refuses_an_empty_folder(segmenter, tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate(segmenter, tmp_path, verbose=False)
