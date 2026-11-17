"""Unit tests for common/plot_style.py."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pytest
from common.plot_style import plot_2d_with_zoom_and_3d_energy, set_ieee_style


def test_plot_style_configuration():
    """Verify IEEE light theme rcParams are applied."""
    set_ieee_style()
    assert plt.rcParams["figure.facecolor"] == "#FFFFFF"
    assert plt.rcParams["axes.facecolor"] == "#FFFFFF"
    assert plt.rcParams["axes.edgecolor"] == "#000000"
    assert plt.rcParams["savefig.dpi"] == 300
    assert "serif" in plt.rcParams["font.family"]


def test_plot_2d_with_zoom_and_3d_energy(tmp_path):
    """Verify publication grid renders both 2D zoomed view and 3D surface plot."""
    img1 = np.full((64, 64), 100.0, dtype=np.float64)
    img1[20:40, 20:40] = 200.0
    img2 = img1 + np.random.RandomState(42).normal(0, 5, img1.shape)

    images_dict = {"Clean": img1, "Degraded": img2}
    save_file = str(tmp_path / "test_pub_figure.png")

    fig, axes = plot_2d_with_zoom_and_3d_energy(
        images_dict=images_dict,
        roi_box=(20, 20, 20, 20),
        save_path=save_file,
        title="IEEE Sensors Journal Test",
    )

    assert fig is not None
    assert axes.shape == (2, 2)
    assert Path(save_file).is_file()
    assert Path(save_file).stat().st_size > 1000
    plt.close(fig)
