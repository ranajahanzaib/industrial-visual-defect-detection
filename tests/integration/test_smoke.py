from pathlib import Path

from industrial_visual_defect_detection.smoke import smoke


def test_smoke_runs_shared_preprocessing_model_heatmap_and_response(tmp_path: Path):
    result = smoke(tmp_path / "synthetic", seed=17, size=64)
    assert result["response_valid"] is True
    assert result["feature_map_shape"] == [64, 64]
    assert "not industrial performance" in result["evidence"]
