import numpy as np

from industrial_visual_defect_detection.evaluation import (
    bootstrap_image_intervals,
    image_metrics,
    pixel_metrics,
    segmentation_metrics,
)


def test_image_metrics_perfect_ranking():
    result = image_metrics([0, 0, 1, 1], [.1, .2, .8, .9], .5)
    assert result["auroc"] == 1.0
    assert result["average_precision"] == 1.0
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]


def test_one_class_auc_is_undefined():
    assert image_metrics([0, 0], [.1, .2], .5)["auroc"] is None


def test_segmentation_metrics():
    result = segmentation_metrics([0, 1, 1, 0], [0, 1, 0, 0])
    assert result == {"dice": 2 / 3, "iou": .5}


def test_empty_masks_are_perfect():
    assert segmentation_metrics([0, 0], [0, 0]) == {"dice": 1.0, "iou": 1.0}


def test_pixel_metrics_use_separate_threshold():
    result = pixel_metrics(np.array([[0, 0], [1, 1]]), np.array([[.1, .2], [.8, .9]]), .5)
    assert result["auroc"] == 1.0
    assert result["average_precision"] == 1.0
    assert result["dice"] == 1.0


def test_bootstrap_is_deterministic_and_resamples_images():
    first = bootstrap_image_intervals([0, 0, 1, 1], [.1, .2, .8, .9], replicates=40, seed=5)
    second = bootstrap_image_intervals([0, 0, 1, 1], [.1, .2, .8, .9], replicates=40, seed=5)
    assert first == second
    assert first["unit"] == "image"
    assert first["auroc_95_ci"][0] <= 1.0 <= first["auroc_95_ci"][1]
