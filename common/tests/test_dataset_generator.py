"""Unit tests for common/dataset_generator.py."""

from pathlib import Path
import numpy as np
import pytest
from common.dataset_generator import (
    BENCHMARK_SCENES,
    generate_benchmark_scenes,
    load_scene,
)


def test_generate_and_load_scenes(tmp_path):
    """Verify all 6 benchmark scenes generate with exact shape (256, 256) and range [0, 255]."""
    scenes = generate_benchmark_scenes(output_dir=tmp_path)

    assert len(scenes) == 6
    for filename in BENCHMARK_SCENES:
        assert filename in scenes
        img = scenes[filename]
        assert isinstance(img, np.ndarray)
        assert img.shape == (256, 256)
        assert img.dtype == np.float64
        assert img.min() >= 0.0
        assert img.max() <= 255.0
        # Scene must contain contrast features (not blank)
        assert np.std(img) >= 5.0

        # Test loading from disk
        loaded = load_scene(filename, clean_scenes_dir=tmp_path)
        assert loaded.shape == (256, 256)
        # Bounded quantization difference due to 8-bit saving
        assert np.max(np.abs(loaded - np.round(img))) <= 1.0


def test_load_scene_missing_raises(tmp_path):
    """load_scene should raise FileNotFoundError for nonexistent non-benchmark files."""
    with pytest.raises(FileNotFoundError):
        load_scene("NonExistentScene_XYZ.png", clean_scenes_dir=tmp_path)
