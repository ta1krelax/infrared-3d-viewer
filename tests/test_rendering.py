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
    SCENE_SURFACE_ZORDER,
    VIEW_PRESETS,
    _SceneMesh,
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
    def test_scene_faces_share_one_depth_sorted_collection(self) -> None:
        renderer = object.__new__(Infrared3DApp)
        figure = Figure(figsize=(4, 3))
        renderer.ax = figure.add_subplot(111, projection="3d", computed_zorder=False)

        coordinates = np.linspace(-1.0, 1.0, 4)
        xx, yy = np.meshgrid(coordinates, coordinates)
        raw = np.full((4, 4), 24.0)
        raw[1:3, 1:3] = 26.0
        display = raw.copy()
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
            below_alpha=0.62,
            water_color=(1.0, 0.0, 0.0),
            water_brightness=1.0,
            water_alpha=0.37,
            gradient_enabled=False,
            gradient_strength=0.0,
            gradient_gamma=1.0,
            show_grid=True,
            grid_count=10,
            show_solid_walls=False,
            show_water_edges=False,
        )

        scene = _SceneMesh()
        sample_base = renderer._draw_sample(
            xx, yy, display, raw, waterline, settings, scene
        )
        renderer._draw_glass_water(
            xx,
            yy,
            waterline,
            raw,
            raw,
            sample_base,
            settings,
            scene,
        )
        renderer.ax.add_collection3d(scene.to_collection())

        # The opaque sample, translucent water and grid must not be separate
        # collections: collection-level painter ordering is what allowed rear
        # water/grid geometry to show through a 100%-opaque front peak.
        self.assertEqual(len(renderer.ax.collections), 1)
        collection = renderer.ax.collections[0]
        self.assertEqual(collection.get_zorder(), SCENE_SURFACE_ZORDER)

        expected_rgb = np.array([1.0, 0.0, 0.0])
        colors = collection.get_facecolors()
        water_colors = colors[np.isclose(colors[:, 3], 0.37)]
        self.assertGreater(len(water_colors), 0)
        np.testing.assert_allclose(
            water_colors[:, :3],
            np.repeat(expected_rgb[None, :], len(water_colors), axis=0),
            atol=1e-12,
        )
        self.assertTrue(np.any(np.isclose(colors[:, 3], 1.0)))
        self.assertTrue(np.any(np.isclose(collection.get_edgecolors()[:, 3], 0.38)))

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
