from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


_SPLIT_RE = re.compile(r"[\s,;]+")


@dataclass(frozen=True)
class TemperatureData:
    values: np.ndarray
    source_format: str


@dataclass(frozen=True)
class ImmersionResult:
    display_height: np.ndarray
    waterline_temperature: float
    submerged_mask: np.ndarray


def _read_text(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw[:256]:
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            pass
    for encoding in ("utf-8-sig", "gb18030", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文本编码。建议将文件另存为 UTF-8。")


def _numeric_rows(text: str) -> list[list[float]]:
    rows: list[list[float]] = []
    for original_line in text.splitlines():
        line = original_line.split("#", 1)[0].split("//", 1)[0].strip()
        if not line:
            continue
        tokens = [token for token in _SPLIT_RE.split(line) if token]
        if not tokens:
            continue
        try:
            row = [float(token) for token in tokens]
        except ValueError:
            # Header and metadata lines are intentionally ignored.
            continue
        rows.append(row)
    return rows


def _largest_rectangular_group(rows: list[list[float]]) -> np.ndarray:
    if not rows:
        raise ValueError("文件中没有找到可读取的数值温度数据。")

    counts: dict[int, int] = {}
    for row in rows:
        counts[len(row)] = counts.get(len(row), 0) + 1
    width = max(counts, key=lambda value: (counts[value] * value, counts[value], value))
    chosen = [row for row in rows if len(row) == width]
    return np.asarray(chosen, dtype=float)


def _try_xyz_grid(array: np.ndarray) -> np.ndarray | None:
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 4:
        return None

    x, y, temperature = array.T
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    if unique_x.size < 2 or unique_y.size < 2:
        return None
    if unique_x.size * unique_y.size != array.shape[0]:
        return None

    x_index = {value: index for index, value in enumerate(unique_x)}
    y_index = {value: index for index, value in enumerate(unique_y)}
    grid = np.full((unique_y.size, unique_x.size), np.nan, dtype=float)
    for x_value, y_value, t_value in zip(x, y, temperature, strict=True):
        row = y_index[y_value]
        column = x_index[x_value]
        if np.isfinite(grid[row, column]):
            return None
        grid[row, column] = t_value
    return grid if np.all(np.isfinite(grid)) else None


def _drop_index_axes(array: np.ndarray) -> np.ndarray:
    result = array
    if result.shape[0] >= 3 and result.shape[1] >= 3:
        first_column = result[:, 0]
        zero_based = np.arange(result.shape[0], dtype=float)
        one_based = zero_based + 1.0
        if np.allclose(first_column, zero_based) or np.allclose(first_column, one_based):
            result = result[:, 1:]

    if result.shape[0] >= 3 and result.shape[1] >= 3:
        first_row = result[0, :]
        zero_based = np.arange(result.shape[1], dtype=float)
        one_based = zero_based + 1.0
        if np.allclose(first_row, zero_based) or np.allclose(first_row, one_based):
            result = result[1:, :]
    return result


def _repair_nonfinite(array: np.ndarray) -> np.ndarray:
    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError("温度数据全部为空值或无穷值。")
    if finite.all():
        return np.asarray(array, dtype=float)

    repaired = np.asarray(array, dtype=float).copy()
    repaired[~finite] = float(np.nanmedian(repaired[finite]))
    return repaired


def load_temperature_txt(path: str | Path) -> TemperatureData:
    """Load either a 2-D temperature matrix or an X/Y/temperature triplet file."""

    rows = _numeric_rows(_read_text(path))
    array = _largest_rectangular_group(rows)

    xyz_grid = _try_xyz_grid(array)
    if xyz_grid is not None:
        values = xyz_grid
        source_format = "XYZ 三列网格"
    else:
        if array.shape[0] == 1 or array.shape[1] == 1:
            flattened = array.ravel()
            side = int(round(np.sqrt(flattened.size)))
            if side * side != flattened.size:
                raise ValueError(
                    "检测到一维数据，但无法推断二维图像尺寸。请将每个像素行写成 TXT 的一行。"
                )
            array = flattened.reshape(side, side)
        values = _drop_index_axes(array)
        source_format = "二维温度矩阵"

    values = _repair_nonfinite(values)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("至少需要 2×2 个温度像素才能绘制表面。")
    return TemperatureData(values=values, source_format=source_format)


def load_temperature_tiff(path: str | Path) -> TemperatureData:
    """Load grayscale TIFF values without reducing their numeric bit depth."""

    try:
        import tifffile
    except ImportError as exc:
        raise ValueError("读取 TIFF 需要 tifffile 组件，请重新安装 requirements.txt。") from exc

    try:
        array = np.asarray(tifffile.imread(path))
    except Exception as exc:
        raise ValueError(f"无法读取 TIFF：{exc}") from exc

    original_shape = array.shape
    original_dtype = array.dtype
    array = np.squeeze(array)
    selected_first_frame = False

    while array.ndim > 3:
        array = array[0]
        selected_first_frame = True

    if array.ndim == 3:
        if array.shape[-1] in {3, 4}:
            rgb = array[..., :3].astype(float)
            array = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        else:
            array = array[0]
            selected_first_frame = True

    if array.ndim != 2:
        raise ValueError(
            f"TIFF 尺寸 {original_shape} 无法转换成二维温度图。"
        )

    values = _repair_nonfinite(np.asarray(array, dtype=float))
    if min(values.shape) < 2:
        raise ValueError("至少需要 2×2 个 TIFF 像素才能绘制表面。")

    bit_depth = original_dtype.itemsize * 8
    type_name = "浮点" if np.issubdtype(original_dtype, np.floating) else "整数"
    frame_note = " · 已取首帧" if selected_first_frame else ""
    source_format = f"TIFF {bit_depth}-bit {type_name}{frame_note}"
    return TemperatureData(values=values, source_format=source_format)


def load_temperature_file(path: str | Path) -> TemperatureData:
    suffix = Path(path).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return load_temperature_tiff(path)
    return load_temperature_txt(path)


def crop_temperature(values: np.ndarray, mode: str = "xy") -> np.ndarray:
    """Crop the center third along both, one, or neither spatial dimension."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("温度数据必须是二维矩阵。")
    if mode not in {"xy", "x", "y", "none"}:
        raise ValueError(f"未知截取方式：{mode}")

    def middle_third(length: int) -> slice:
        if length < 3:
            return slice(0, length)
        start = length // 3
        stop = length - start
        return slice(start, stop)

    row_slice = middle_third(array.shape[0]) if mode in {"xy", "y"} else slice(None)
    column_slice = middle_third(array.shape[1]) if mode in {"xy", "x"} else slice(None)
    result = array[row_slice, column_slice]
    if min(result.shape) < 2:
        raise ValueError("截取后数据太小，无法绘制 3D 表面。")
    return result


def save_cropped_source_copy(
    values: np.ndarray,
    source_name: str | Path,
    image_output: str | Path,
) -> Path:
    """Save cropped source values beside an exported image without overwriting."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("导出的裁剪源数据必须是非空、有限的二维矩阵。")

    source_suffix = Path(source_name).suffix.lower()
    if source_suffix not in {".txt", ".csv", ".dat", ".tif", ".tiff"}:
        source_suffix = ".txt"

    image_path = Path(image_output)
    base = image_path.with_name(f"{image_path.stem}_cropped_source{source_suffix}")
    output = base
    copy_number = 2
    while output.exists():
        output = base.with_name(f"{base.stem}_{copy_number}{base.suffix}")
        copy_number += 1

    if source_suffix in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError as exc:
            raise ValueError("导出 TIFF 数据副本需要 tifffile 组件。") from exc
        tifffile.imwrite(
            output,
            array.astype(np.float32),
            compression="lzw",
            metadata=None,
        )
    else:
        delimiter = "," if source_suffix == ".csv" else "\t"
        np.savetxt(output, array, fmt="%.12g", delimiter=delimiter)
    return output


def apply_immersion_absorption(
    temperature: np.ndarray,
    immersion_ratio: float,
    absorption_ratio: float,
) -> ImmersionResult:
    """Compress submerged height differences by (1 - absorption_ratio)."""

    values = np.asarray(temperature, dtype=float)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("温度数据必须是非空二维矩阵。")
    if not 0.0 <= immersion_ratio <= 1.0:
        raise ValueError("浸泡比例必须位于 0 到 1 之间。")
    if not 0.0 <= absorption_ratio <= 1.0:
        raise ValueError("吸收比例必须位于 0 到 1 之间。")

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    waterline = minimum + immersion_ratio * (maximum - minimum)
    submerged = values < waterline
    display_height = values.copy()
    display_height[submerged] = waterline + (
        values[submerged] - waterline
    ) * (1.0 - absorption_ratio)
    return ImmersionResult(
        display_height=display_height,
        waterline_temperature=waterline,
        submerged_mask=submerged,
    )


def refine_temperature_grid(
    temperature: np.ndarray,
    waterline: float,
    levels: int = 1,
) -> np.ndarray:
    """Insert midpoint samples while pinning waterline crossings to the threshold."""

    values = np.asarray(temperature, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("细化至少需要 2×2 个数据点。")
    if levels < 0:
        raise ValueError("细化次数不能为负数。")

    def refine_axis(array: np.ndarray, axis: int) -> np.ndarray:
        moved = np.moveaxis(array, axis, 0)
        refined_shape = (moved.shape[0] * 2 - 1, *moved.shape[1:])
        refined = np.empty(refined_shape, dtype=float)
        refined[0::2] = moved
        first = moved[:-1]
        second = moved[1:]
        crosses = ((first < waterline) & (second > waterline)) | (
            (first > waterline) & (second < waterline)
        )
        midpoint = 0.5 * (first + second)
        refined[1::2] = np.where(crosses, waterline, midpoint)
        return np.moveaxis(refined, 0, axis)

    refined = values.copy()
    for _ in range(levels):
        refined = refine_axis(refined, axis=1)
        refined = refine_axis(refined, axis=0)
    return refined


def downsample_grid(
    values: np.ndarray, max_rows: int = 180, max_columns: int = 180
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a display-sized array plus the selected source row/column indices."""

    array = np.asarray(values, dtype=float)
    row_indices = np.unique(
        np.linspace(0, array.shape[0] - 1, min(array.shape[0], max_rows)).round().astype(int)
    )
    column_indices = np.unique(
        np.linspace(0, array.shape[1] - 1, min(array.shape[1], max_columns))
        .round()
        .astype(int)
    )
    return array[np.ix_(row_indices, column_indices)], row_indices, column_indices


def make_demo_data(rows: int = 120, columns: int = 168) -> np.ndarray:
    """Create a deterministic sample with several thermal peaks."""

    y, x = np.mgrid[-1.0:1.0:complex(rows), -1.35:1.35:complex(columns)]
    broad = 1.35 * np.exp(-((x + 0.05) ** 2 / 0.48 + (y + 0.02) ** 2 / 0.30))
    peak = 0.78 * np.exp(-((x - 0.42) ** 2 / 0.045 + (y + 0.12) ** 2 / 0.07))
    shoulder = 0.36 * np.exp(-((x + 0.55) ** 2 / 0.13 + (y - 0.28) ** 2 / 0.11))
    texture = 0.035 * np.sin(18.0 * x + 3.0 * y) * np.cos(13.0 * y)
    return 24.0 + broad + peak + shoulder + texture
