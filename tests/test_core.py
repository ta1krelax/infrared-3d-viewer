from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from core import (  # noqa: E402
    apply_immersion_absorption,
    crop_temperature,
    downsample_grid,
    load_temperature_file,
    load_temperature_tiff,
    load_temperature_txt,
    refine_temperature_grid,
)


class CoreTests(unittest.TestCase):
    def test_center_third_both_axes(self) -> None:
        source = np.arange(9 * 12, dtype=float).reshape(9, 12)
        cropped = crop_temperature(source, "xy")
        np.testing.assert_array_equal(cropped, source[3:6, 4:8])

    def test_absorption_matches_requested_example(self) -> None:
        source = np.array([[24.0, 25.8, 26.0]])
        result = apply_immersion_absorption(source, 0.9, 0.8)
        self.assertAlmostEqual(result.waterline_temperature, 25.8)
        np.testing.assert_allclose(result.display_height, [[25.44, 25.8, 26.0]])
        np.testing.assert_array_equal(result.submerged_mask, [[True, False, False]])

    def test_matrix_txt_with_header(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "matrix.txt"
            path.write_text("IR temperature frame\n24.0 24.2 24.3\n24.1 25.0 25.2\n", encoding="utf-8")
            loaded = load_temperature_txt(path)
        self.assertEqual(loaded.source_format, "二维温度矩阵")
        np.testing.assert_allclose(loaded.values, [[24.0, 24.2, 24.3], [24.1, 25.0, 25.2]])

    def test_xyz_txt(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "xyz.txt"
            path.write_text("0,0,24\n1,0,25\n0,1,26\n1,1,27\n", encoding="utf-8")
            loaded = load_temperature_txt(path)
        self.assertEqual(loaded.source_format, "XYZ 三列网格")
        np.testing.assert_allclose(loaded.values, [[24, 25], [26, 27]])

    def test_utf16_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "utf16.txt"
            path.write_text("24.0\t24.5\n25.0\t25.5\n", encoding="utf-16")
            loaded = load_temperature_txt(path)
        np.testing.assert_allclose(loaded.values, [[24.0, 24.5], [25.0, 25.5]])

    def test_tiff_numeric_formats_preserve_values(self) -> None:
        import tifffile

        with tempfile.TemporaryDirectory() as folder:
            for dtype in (np.uint8, np.uint16, np.float16, np.float32):
                with self.subTest(dtype=np.dtype(dtype).name):
                    source = np.linspace(1, 12, 12, dtype=dtype).reshape(3, 4)
                    path = Path(folder) / f"sample_{np.dtype(dtype).name}.tif"
                    tifffile.imwrite(path, source)
                    loaded = load_temperature_tiff(path)
                    np.testing.assert_allclose(loaded.values, source.astype(float))
                    self.assertIn(str(np.dtype(dtype).itemsize * 8), loaded.source_format)

    def test_dispatches_tiff_by_extension(self) -> None:
        import tifffile

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.tiff"
            tifffile.imwrite(path, np.arange(9, dtype=np.float32).reshape(3, 3))
            loaded = load_temperature_file(path)
        self.assertIn("TIFF", loaded.source_format)

    def test_compressed_uint16_tiff(self) -> None:
        import tifffile

        with tempfile.TemporaryDirectory() as folder:
            source = np.arange(20, dtype=np.uint16).reshape(4, 5)
            path = Path(folder) / "compressed.tif"
            tifffile.imwrite(path, source, compression="lzw")
            loaded = load_temperature_tiff(path)
        np.testing.assert_allclose(loaded.values, source)

    def test_downsample_keeps_corners(self) -> None:
        source = np.arange(100 * 120).reshape(100, 120)
        reduced, rows, columns = downsample_grid(source, 12, 14)
        self.assertLessEqual(reduced.shape[0], 12)
        self.assertLessEqual(reduced.shape[1], 14)
        self.assertEqual(rows[0], 0)
        self.assertEqual(rows[-1], 99)
        self.assertEqual(columns[0], 0)
        self.assertEqual(columns[-1], 119)

    def test_refinement_inserts_exact_waterline_at_crossing(self) -> None:
        source = np.array([[24.0, 26.0], [24.0, 26.0]])
        refined = refine_temperature_grid(source, waterline=25.0)
        np.testing.assert_allclose(
            refined,
            [[24.0, 25.0, 26.0], [24.0, 25.0, 26.0], [24.0, 25.0, 26.0]],
        )

    def test_refinement_averages_non_crossing_neighbors(self) -> None:
        source = np.array([[24.0, 24.5], [24.4, 24.8]])
        refined = refine_temperature_grid(source, waterline=25.0)
        self.assertAlmostEqual(refined[0, 1], 24.25)
        self.assertAlmostEqual(refined[1, 0], 24.20)
        self.assertAlmostEqual(refined[1, 1], 24.425)

    def test_repeated_refinement_preserves_range_and_shape(self) -> None:
        source = np.array([[24.0, 26.0], [25.0, 25.5]])
        refined = refine_temperature_grid(source, waterline=25.4, levels=3)
        self.assertEqual(refined.shape, (9, 9))
        self.assertEqual(float(np.min(refined)), 24.0)
        self.assertEqual(float(np.max(refined)), 26.0)


if __name__ == "__main__":
    unittest.main()
