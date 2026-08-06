from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
from matplotlib.figure import Figure


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from app import (  # noqa: E402
    DEFAULT_VIEW_NAME,
    Infrared3DApp,
    RenderSettings,
    SAMPLE_SURFACE_ZORDER,
    VIEW_PRESETS,
    WATER_BOTTOM_ZORDER,
    WATER_SIDE_ZORDER,
    WATER_TOP_ZORDER,
)


class _VariableStub:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


class _CanvasStub:
    def __init__(self) -> None:
        self.draw_count = 0

    def draw_idle(self) -> None:
        self.draw_count += 1


class WaterRenderingTests(unittest.TestCase):
    def test_water_faces_share_material_and_render_after_sample(self) -> None:
        renderer = object.__new__(Infrared3DApp)
        figure = Figure(figsize=(4, 3))
        renderer.ax = figure.add_subplot(111, projection="3d", computed_zorder=False)

        coordinates = np.linspace(-1.0, 1.0, 3)
        xx, yy = np.meshgrid(coordinates, coordinates)
        raw = np.full((3, 3), 24.0)
        display = np.full((3, 3), 24.6)
        waterline = 25.0
        settings = RenderSettings(
            crop_mode="none",
            pixel_size=0.01,
            immersion=0.5,
            absorption=0.8,
            x_scale=1.0,
            y_scale=1.0,
            vertical_scale=1.0,
            above_color=(0.8, 0.8, 0.8),
            above_brightness=1.0,
            above_alpha=1.0,
            below_color=(0.4, 0.4, 0.4),
            below_brightness=1.0,
            below_alpha=1.0,
            water_color=(1.0, 0.0, 0.0),
            water_brightness=1.0,
            water_alpha=0.37,
            gradient_enabled=False,
            gradient_strength=0.0,
            gradient_gamma=1.0,
            show_grid=False,
            grid_count=10,
            show_solid_walls=False,
            show_water_edges=False,
        )

        sample_base = renderer._draw_sample(
            xx, yy, display, raw, waterline, settings
        )
        renderer._draw_glass_water(
            xx,
            yy,
            waterline,
            raw,
            raw,
            sample_base,
            settings,
        )

        collections_by_zorder = {
            collection.get_zorder(): collection for collection in renderer.ax.collections
        }
        self.assertIn(SAMPLE_SURFACE_ZORDER, collections_by_zorder)
        self.assertIn(WATER_BOTTOM_ZORDER, collections_by_zorder)
        self.assertIn(WATER_SIDE_ZORDER, collections_by_zorder)
        self.assertIn(WATER_TOP_ZORDER, collections_by_zorder)
        self.assertLess(SAMPLE_SURFACE_ZORDER, WATER_BOTTOM_ZORDER)
        self.assertLess(WATER_BOTTOM_ZORDER, WATER_SIDE_ZORDER)
        self.assertLess(WATER_SIDE_ZORDER, WATER_TOP_ZORDER)

        expected_rgb = np.array([1.0, 0.0, 0.0])
        for zorder in (WATER_BOTTOM_ZORDER, WATER_SIDE_ZORDER, WATER_TOP_ZORDER):
            colors = collections_by_zorder[zorder].get_facecolors()
            expected_colors = np.repeat(expected_rgb[None, :], len(colors), axis=0)
            np.testing.assert_allclose(colors[:, :3], expected_colors, atol=1e-12)
            np.testing.assert_allclose(colors[:, 3], 0.37, atol=1e-12)

    def test_view_presets_apply_angles_without_rebuilding_plot(self) -> None:
        renderer = object.__new__(Infrared3DApp)
        figure = Figure(figsize=(4, 3))
        renderer.ax = figure.add_subplot(111, projection="3d")
        renderer.canvas = _CanvasStub()
        renderer.status_var = _VariableStub()
        renderer.view_preset_var = _VariableStub("立方体 · 角（等轴测）")

        renderer.apply_view_preset()

        self.assertAlmostEqual(renderer.ax.elev, 35.264)
        self.assertAlmostEqual(renderer.ax.azim, -45.0)
        self.assertEqual(renderer.canvas.draw_count, 1)
        self.assertIn("等轴测", renderer.status_var.get())

        renderer.reset_view()
        self.assertEqual(renderer.view_preset_var.get(), DEFAULT_VIEW_NAME)
        self.assertEqual(
            (renderer.ax.elev, renderer.ax.azim, renderer.ax.roll),
            VIEW_PRESETS[DEFAULT_VIEW_NAME],
        )


if __name__ == "__main__":
    unittest.main()
