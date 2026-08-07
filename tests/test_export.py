from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from app import save_figure_image  # noqa: E402
from core import save_cropped_source_copy  # noqa: E402


class TransparentExportTests(unittest.TestCase):
    @staticmethod
    def _make_figure() -> Figure:
        figure = Figure(figsize=(2, 2), dpi=80, facecolor="#f4f6f8")
        axes = figure.add_axes((0.1, 0.1, 0.8, 0.8))
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        axes.set_axis_off()
        axes.add_patch(
            Rectangle((0.3, 0.3), 0.4, 0.4, facecolor="#d62728", edgecolor="none")
        )
        return figure

    def test_png_and_tiff_export_have_true_alpha_background(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            for suffix in (".png", ".tif"):
                with self.subTest(suffix=suffix):
                    output = Path(folder) / f"transparent{suffix}"
                    save_figure_image(
                        self._make_figure(), output, dpi=80, transparent_background=True
                    )
                    with Image.open(output) as image:
                        alpha = image.convert("RGBA").getchannel("A")
                        alpha_min, alpha_max = alpha.getextrema()
                    self.assertEqual(alpha_min, 0)
                    self.assertEqual(alpha_max, 255)

    def test_background_can_still_be_exported_when_option_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "opaque.png"
            save_figure_image(
                self._make_figure(), output, dpi=80, transparent_background=False
            )
            with Image.open(output) as image:
                alpha_min, alpha_max = image.convert("RGBA").getchannel("A").getextrema()
            self.assertEqual((alpha_min, alpha_max), (255, 255))

    def test_cropped_text_copy_uses_image_name_and_never_overwrites(self) -> None:
        values = np.array([[24.125, 25.25], [26.5, 27.75]])
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "figure.png"
            first = save_cropped_source_copy(values, "camera.csv", image)
            second = save_cropped_source_copy(values, "camera.csv", image)

            self.assertEqual(first.name, "figure_cropped_source.csv")
            self.assertEqual(second.name, "figure_cropped_source_2.csv")
            np.testing.assert_allclose(np.loadtxt(first, delimiter=","), values)
            np.testing.assert_allclose(np.loadtxt(second, delimiter=","), values)

    def test_cropped_tiff_copy_is_reloadable_float32(self) -> None:
        import tifffile

        values = np.array([[24.125, 25.25], [26.5, 27.75]])
        with tempfile.TemporaryDirectory() as folder:
            output = save_cropped_source_copy(
                values,
                "camera.tiff",
                Path(folder) / "figure.png",
            )
            restored = tifffile.imread(output)

            self.assertEqual(output.name, "figure_cropped_source.tiff")
            self.assertEqual(restored.dtype, np.float32)
            np.testing.assert_allclose(restored, values)


if __name__ == "__main__":
    unittest.main()
