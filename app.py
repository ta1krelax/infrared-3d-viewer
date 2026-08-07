from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LightSource, to_rgb
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from core import (
    apply_immersion_absorption,
    crop_bounds_for_mode,
    crop_temperature,
    downsample_grid,
    load_temperature_file,
    make_demo_data,
    refine_temperature_grid,
    save_cropped_source_copy,
    validate_crop_bounds,
)


APP_TITLE = "红外温度 · 3D 浸泡示意图"
DEFAULT_CROP_LABEL = "中央 1/3（X、Y）"
CUSTOM_CROP_LABEL = "自定义区域"
CROP_LABELS = {
    DEFAULT_CROP_LABEL: "xy",
    "仅 X 方向中央 1/3": "x",
    "仅 Y 方向中央 1/3": "y",
    "不截取": "none",
    CUSTOM_CROP_LABEL: "custom",
}
MAX_REFINEMENT_LEVEL = 6
MAX_REFINED_POINTS = 90_000

DEFAULT_VIEW_NAME = "常规透视"
CUSTOM_VIEW_NAME = "自定义数值"
PERSPECTIVE_LABEL = "透视投影"
ORTHOGRAPHIC_LABEL = "正交投影"
PROJECTION_MODES = {
    PERSPECTIVE_LABEL: "persp",
    ORTHOGRAPHIC_LABEL: "ortho",
}


@dataclass(frozen=True)
class ViewPreset:
    projection: str
    rotation_x: float
    rotation_y: float
    rotation_z: float = 0.0
    zoom_percent: float = 100.0


# The numeric controls describe sample rotation instead of Matplotlib's camera
# angles.  A zero rotation is the front face; horizontal sample rotation Y is
# camera azimuth + 90 degrees.
VIEW_PRESETS: dict[str, ViewPreset] = {
    DEFAULT_VIEW_NAME: ViewPreset(PERSPECTIVE_LABEL, 28.0, 35.0),
    "立方体 · 面（正视）": ViewPreset(PERSPECTIVE_LABEL, 0.0, 0.0),
    "立方体 · 棱（双面）": ViewPreset(PERSPECTIVE_LABEL, 0.0, 45.0),
    "立方体 · 角（等轴测）": ViewPreset(PERSPECTIVE_LABEL, 35.264, 45.0),
    "反向等轴测": ViewPreset(PERSPECTIVE_LABEL, 35.264, -135.0),
    "俯视（XY）": ViewPreset(PERSPECTIVE_LABEL, 90.0, 0.0),
    "仰视（XY）": ViewPreset(PERSPECTIVE_LABEL, -90.0, 0.0),
    "后视": ViewPreset(PERSPECTIVE_LABEL, 0.0, 180.0),
    "左视": ViewPreset(PERSPECTIVE_LABEL, 0.0, 90.0),
    "右视": ViewPreset(PERSPECTIVE_LABEL, 0.0, -90.0),
    "高角度透视": ViewPreset(PERSPECTIVE_LABEL, 55.0, 45.0),
    "低角度透视": ViewPreset(PERSPECTIVE_LABEL, 15.0, 35.0),
    "正交 · 正视": ViewPreset(ORTHOGRAPHIC_LABEL, 0.0, 0.0),
    "正交 · 后视": ViewPreset(ORTHOGRAPHIC_LABEL, 0.0, 180.0),
    "正交 · 左视": ViewPreset(ORTHOGRAPHIC_LABEL, 0.0, 90.0),
    "正交 · 右视": ViewPreset(ORTHOGRAPHIC_LABEL, 0.0, -90.0),
    "正交 · 俯视": ViewPreset(ORTHOGRAPHIC_LABEL, 90.0, 0.0),
    "正交 · 等轴测": ViewPreset(ORTHOGRAPHIC_LABEL, 35.264, 45.0),
    "正交 · 反向等轴测": ViewPreset(ORTHOGRAPHIC_LABEL, 35.264, -135.0),
}


def sample_rotation_to_camera(
    rotation_x: float, rotation_y: float, rotation_z: float
) -> tuple[float, float, float]:
    return rotation_x, rotation_y - 90.0, rotation_z


def _normalize_angle(angle: float) -> float:
    normalized = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if np.isclose(normalized, -180.0) and angle > 0 else normalized


def camera_to_sample_rotation(
    elevation: float, azimuth: float, roll: float
) -> tuple[float, float, float]:
    return (
        _normalize_angle(elevation),
        _normalize_angle(azimuth + 90.0),
        _normalize_angle(roll),
    )

# Matplotlib does not have a hardware depth buffer for its 3-D artists.  If the
# sample, grid and water are separate collections, it can only sort those whole
# collections and a rear translucent surface may be painted over a front
# opaque peak.  All scene polygons therefore live in one collection so they
# are depth-sorted face by face for every camera angle.
SCENE_SURFACE_ZORDER = 10


class _SceneMesh:
    """Collect surfaces and line segments into one depth-sorted 3-D artist."""

    def __init__(self) -> None:
        self._faces: list[np.ndarray] = []
        self._facecolors: list[np.ndarray] = []
        self._edgecolors: list[np.ndarray] = []

    def add_faces(
        self,
        faces: np.ndarray | list[list[tuple[float, float, float]]],
        facecolors: np.ndarray | tuple[float, float, float, float],
        edgecolors: np.ndarray | tuple[float, float, float, float] = (0, 0, 0, 0),
    ) -> None:
        face_array = np.asarray(faces, dtype=float)
        if face_array.size == 0:
            return
        if face_array.ndim != 3 or face_array.shape[1:] != (4, 3):
            raise ValueError("3-D scene faces must have shape (n, 4, 3).")

        count = face_array.shape[0]
        face_color_array = np.asarray(facecolors, dtype=float)
        if face_color_array.ndim == 1:
            face_color_array = np.broadcast_to(face_color_array, (count, 4)).copy()
        edge_color_array = np.asarray(edgecolors, dtype=float)
        if edge_color_array.ndim == 1:
            edge_color_array = np.broadcast_to(edge_color_array, (count, 4)).copy()
        if face_color_array.shape != (count, 4) or edge_color_array.shape != (count, 4):
            raise ValueError("Every scene face must have one RGBA face and edge color.")

        self._faces.append(face_array)
        self._facecolors.append(face_color_array)
        self._edgecolors.append(edge_color_array)

    def add_surface(
        self,
        xx: np.ndarray,
        yy: np.ndarray,
        zz: np.ndarray,
        vertex_colors: np.ndarray | tuple[float, float, float, float],
    ) -> None:
        """Append valid grid cells as quads, averaging their vertex colors."""
        points = np.stack((xx, yy, zz), axis=-1)
        faces = np.stack(
            (
                points[:-1, :-1],
                points[:-1, 1:],
                points[1:, 1:],
                points[1:, :-1],
            ),
            axis=2,
        )
        valid = np.all(np.isfinite(faces), axis=(2, 3))
        if not np.any(valid):
            return

        colors = np.asarray(vertex_colors, dtype=float)
        if colors.ndim == 1:
            cell_colors = np.broadcast_to(colors, (*valid.shape, 4))
        else:
            cell_colors = 0.25 * (
                colors[:-1, :-1]
                + colors[:-1, 1:]
                + colors[1:, 1:]
                + colors[1:, :-1]
            )
        self.add_faces(faces[valid], cell_colors[valid])

    def add_polyline(
        self,
        x: np.ndarray | list[float],
        y: np.ndarray | list[float],
        z: np.ndarray | list[float],
        color: tuple[float, float, float, float],
    ) -> None:
        """Append each line segment as a degenerate quad for shared sorting."""
        points = np.column_stack((x, y, z)).astype(float, copy=False)
        if len(points) < 2:
            return
        valid = np.all(np.isfinite(points[:-1]), axis=1) & np.all(
            np.isfinite(points[1:]), axis=1
        )
        if not np.any(valid):
            return
        start = points[:-1][valid]
        end = points[1:][valid]
        faces = np.stack((start, end, end, start), axis=1)
        self.add_faces(faces, (0, 0, 0, 0), color)

    def to_collection(self) -> Poly3DCollection:
        if not self._faces:
            raise ValueError("The 3-D scene contains no faces.")
        return Poly3DCollection(
            np.concatenate(self._faces, axis=0),
            facecolors=np.concatenate(self._facecolors, axis=0),
            edgecolors=np.concatenate(self._edgecolors, axis=0),
            linewidths=0.42,
            antialiaseds=False,
            zsort="average",
            zorder=SCENE_SURFACE_ZORDER,
        )


@dataclass(frozen=True)
class RenderSettings:
    crop_mode: str
    pixel_size: float
    immersion: float
    absorption: float
    x_scale: float
    y_scale: float
    vertical_scale: float
    above_color: tuple[float, float, float]
    above_brightness: float
    above_alpha: float
    below_color: tuple[float, float, float]
    below_brightness: float
    below_alpha: float
    water_color: tuple[float, float, float]
    water_brightness: float
    water_alpha: float
    gradient_enabled: bool
    gradient_strength: float
    gradient_gamma: float
    show_grid: bool
    grid_count: int
    show_solid_walls: bool
    show_water_edges: bool


def _adjust_rgb(rgb: tuple[float, float, float] | np.ndarray, brightness: float) -> np.ndarray:
    return np.clip(np.asarray(rgb, dtype=float) * brightness, 0.0, 1.0)


def save_figure_image(
    figure: Figure,
    selected: str | Path,
    dpi: int,
    transparent_background: bool,
) -> Path:
    """Save the current view while preserving RGBA transparency when requested."""
    output = Path(selected)
    suffix = output.suffix.lower()
    if suffix not in {".png", ".tif", ".tiff"}:
        output = Path(f"{output}.png")
        suffix = ".png"

    save_options: dict[str, object] = {
        "dpi": dpi,
        "bbox_inches": "tight",
        "pad_inches": 0.02,
    }
    if transparent_background:
        # `transparent=True` temporarily makes both the Figure and Axes patches
        # transparent, while retaining every artist's own alpha value.
        save_options.update(
            transparent=True,
            facecolor="none",
            edgecolor="none",
        )
    else:
        save_options["facecolor"] = figure.get_facecolor()

    if suffix in {".tif", ".tiff"}:
        save_options["format"] = "tiff"
        save_options["pil_kwargs"] = {"compression": "tiff_lzw"}
    else:
        save_options["format"] = "png"
    figure.savefig(output, **save_options)
    return output


class CropSelectionDialog(tk.Toplevel):
    """Modal original-data preview with a movable, resizable pixel crop box."""

    PREVIEW_WIDTH = 760
    PREVIEW_HEIGHT = 480
    HANDLE_RADIUS = 5
    MIN_SIZE = 2

    def __init__(
        self,
        parent: tk.Misc,
        values: np.ndarray,
        initial_bounds: tuple[int, int, int, int],
    ) -> None:
        super().__init__(parent)
        self.title("选择裁剪区域")
        self.configure(background="#f7f8fa")
        self.resizable(False, False)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self.values = np.asarray(values, dtype=float)
        self.rows, self.columns = self.values.shape
        row_start, row_stop, column_start, column_stop = validate_crop_bounds(
            initial_bounds, self.values.shape
        )
        self.bounds = [column_start, row_start, column_stop, row_stop]
        self.result: tuple[int, int, int, int] | None = None
        self._drag_mode: str | None = None
        self._drag_start = (0, 0)
        self._drag_bounds = self.bounds.copy()
        self._syncing_numeric = False

        outer = ttk.Frame(self, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            outer,
            text="拖动框内区域可移动；拖动八个控制点可改变尺寸；在框外拖动可重新框选。",
            wraplength=self.PREVIEW_WIDTH,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(0, 8))

        self.canvas = tk.Canvas(
            outer,
            width=self.PREVIEW_WIDTH,
            height=self.PREVIEW_HEIGHT,
            background="#20252b",
            highlightthickness=1,
            highlightbackground="#aab4be",
            cursor="crosshair",
        )
        self.canvas.pack()
        self._build_preview_image()
        self.selection_id = self.canvas.create_rectangle(
            0, 0, 1, 1, outline="#00e5ff", width=2
        )
        self.handle_ids = {
            name: self.canvas.create_rectangle(
                0,
                0,
                1,
                1,
                fill="#ffffff",
                outline="#007c91",
                width=1,
            )
            for name in ("nw", "n", "ne", "e", "se", "s", "sw", "w")
        }
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        numeric = ttk.LabelFrame(outer, text="像素裁剪范围", padding=(10, 8))
        numeric.pack(fill=tk.X, pady=(10, 0))
        self.x_var = tk.IntVar()
        self.y_var = tk.IntVar()
        self.width_var = tk.IntVar()
        self.height_var = tk.IntVar()
        fields = (
            ("X", self.x_var, 0, max(0, self.columns - self.MIN_SIZE)),
            ("Y", self.y_var, 0, max(0, self.rows - self.MIN_SIZE)),
            ("宽度", self.width_var, self.MIN_SIZE, self.columns),
            ("高度", self.height_var, self.MIN_SIZE, self.rows),
        )
        for column, (label, variable, minimum, maximum) in enumerate(fields):
            ttk.Label(numeric, text=label).grid(row=0, column=column * 2, padx=(0, 4))
            spinbox = ttk.Spinbox(
                numeric,
                textvariable=variable,
                from_=minimum,
                to=maximum,
                increment=1,
                width=8,
                command=self._apply_numeric,
            )
            spinbox.grid(row=0, column=column * 2 + 1, padx=(0, 12))
            spinbox.bind("<Return>", self._apply_numeric)
            spinbox.bind("<FocusOut>", self._apply_numeric)
        numeric.columnconfigure(8, weight=1)
        self.size_label = ttk.Label(numeric)
        self.size_label.grid(row=0, column=8, sticky="e")

        buttons = ttk.Frame(outer)
        buttons.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(buttons, text="全图", command=self._select_full).pack(side=tk.LEFT)
        ttk.Button(buttons, text="中央 1/3", command=self._select_middle).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(buttons, text="取消", command=self._cancel).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="应用裁剪", command=self._confirm).pack(
            side=tk.RIGHT, padx=(0, 8)
        )

        self._draw_selection()
        self.update_idletasks()
        parent_x = parent.winfo_rootx()
        parent_y = parent.winfo_rooty()
        x = parent_x + max(0, (parent.winfo_width() - self.winfo_width()) // 2)
        y = parent_y + max(0, (parent.winfo_height() - self.winfo_height()) // 2)
        self.geometry(f"+{x}+{y}")
        self.grab_set()

    def _build_preview_image(self) -> None:
        finite = self.values[np.isfinite(self.values)]
        low, high = np.percentile(finite, (1.0, 99.0))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.min(finite))
            high = float(np.max(finite))
        span = max(high - low, 1e-12)
        normalized = np.clip((self.values - low) / span, 0.0, 1.0)
        rgb = matplotlib.colormaps["inferno"](normalized, bytes=True)[..., :3]
        preview = Image.fromarray(rgb, mode="RGB")
        scale = min(
            self.PREVIEW_WIDTH / self.columns,
            self.PREVIEW_HEIGHT / self.rows,
        )
        display_width = max(1, int(round(self.columns * scale)))
        display_height = max(1, int(round(self.rows * scale)))
        preview = preview.resize(
            (display_width, display_height), Image.Resampling.BILINEAR
        )
        self.scale_x = display_width / self.columns
        self.scale_y = display_height / self.rows
        self.origin_x = 0.5 * (self.PREVIEW_WIDTH - display_width)
        self.origin_y = 0.5 * (self.PREVIEW_HEIGHT - display_height)
        self.preview_photo = ImageTk.PhotoImage(preview, master=self)
        self.canvas.create_image(
            self.origin_x,
            self.origin_y,
            image=self.preview_photo,
            anchor="nw",
        )

    def _image_to_canvas(self, x: int, y: int) -> tuple[float, float]:
        return self.origin_x + x * self.scale_x, self.origin_y + y * self.scale_y

    def _canvas_to_image(self, x: float, y: float) -> tuple[int, int]:
        image_x = int(round((x - self.origin_x) / self.scale_x))
        image_y = int(round((y - self.origin_y) / self.scale_y))
        return (
            min(self.columns, max(0, image_x)),
            min(self.rows, max(0, image_y)),
        )

    def _draw_selection(self) -> None:
        x0, y0, x1, y1 = self.bounds
        left, top = self._image_to_canvas(x0, y0)
        right, bottom = self._image_to_canvas(x1, y1)
        self.canvas.coords(self.selection_id, left, top, right, bottom)
        positions = {
            "nw": (left, top),
            "n": ((left + right) / 2, top),
            "ne": (right, top),
            "e": (right, (top + bottom) / 2),
            "se": (right, bottom),
            "s": ((left + right) / 2, bottom),
            "sw": (left, bottom),
            "w": (left, (top + bottom) / 2),
        }
        radius = self.HANDLE_RADIUS
        for name, (center_x, center_y) in positions.items():
            self.canvas.coords(
                self.handle_ids[name],
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            )
        self._sync_numeric()

    def _sync_numeric(self) -> None:
        self._syncing_numeric = True
        x0, y0, x1, y1 = self.bounds
        self.x_var.set(x0)
        self.y_var.set(y0)
        self.width_var.set(x1 - x0)
        self.height_var.set(y1 - y0)
        self.size_label.configure(
            text=f"原图 {self.columns} × {self.rows} px"
        )
        self._syncing_numeric = False

    def _apply_numeric(self, _event: tk.Event | None = None) -> None:
        if self._syncing_numeric:
            return
        try:
            x0 = int(self.x_var.get())
            y0 = int(self.y_var.get())
            width = int(self.width_var.get())
            height = int(self.height_var.get())
        except (ValueError, tk.TclError):
            return
        width = min(self.columns, max(self.MIN_SIZE, width))
        height = min(self.rows, max(self.MIN_SIZE, height))
        x0 = min(self.columns - width, max(0, x0))
        y0 = min(self.rows - height, max(0, y0))
        self.bounds = [x0, y0, x0 + width, y0 + height]
        self._draw_selection()

    def _hit_handle(self, canvas_x: float, canvas_y: float) -> str | None:
        tolerance = self.HANDLE_RADIUS + 3
        for name, item_id in self.handle_ids.items():
            left, top, right, bottom = self.canvas.coords(item_id)
            center_x = 0.5 * (left + right)
            center_y = 0.5 * (top + bottom)
            if abs(canvas_x - center_x) <= tolerance and abs(canvas_y - center_y) <= tolerance:
                return name
        return None

    def _on_press(self, event: tk.Event) -> None:
        image_x, image_y = self._canvas_to_image(event.x, event.y)
        self._drag_start = (image_x, image_y)
        self._drag_bounds = self.bounds.copy()
        handle = self._hit_handle(event.x, event.y)
        if handle is not None:
            self._drag_mode = handle
            return
        x0, y0, x1, y1 = self.bounds
        if x0 <= image_x <= x1 and y0 <= image_y <= y1:
            self._drag_mode = "move"
        else:
            self._drag_mode = "new"

    @staticmethod
    def _minimum_interval(start: int, stop: int, limit: int) -> tuple[int, int]:
        low, high = sorted((start, stop))
        if high - low >= CropSelectionDialog.MIN_SIZE:
            return low, high
        high = min(limit, low + CropSelectionDialog.MIN_SIZE)
        low = max(0, high - CropSelectionDialog.MIN_SIZE)
        return low, high

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag_mode is None:
            return
        image_x, image_y = self._canvas_to_image(event.x, event.y)
        start_x, start_y = self._drag_start
        x0, y0, x1, y1 = self._drag_bounds
        if self._drag_mode == "move":
            width, height = x1 - x0, y1 - y0
            new_x0 = min(self.columns - width, max(0, x0 + image_x - start_x))
            new_y0 = min(self.rows - height, max(0, y0 + image_y - start_y))
            self.bounds = [new_x0, new_y0, new_x0 + width, new_y0 + height]
        elif self._drag_mode == "new":
            new_x0, new_x1 = self._minimum_interval(start_x, image_x, self.columns)
            new_y0, new_y1 = self._minimum_interval(start_y, image_y, self.rows)
            self.bounds = [new_x0, new_y0, new_x1, new_y1]
        else:
            if "w" in self._drag_mode:
                x0 = min(image_x, x1 - self.MIN_SIZE)
            if "e" in self._drag_mode:
                x1 = max(image_x, x0 + self.MIN_SIZE)
            if "n" in self._drag_mode:
                y0 = min(image_y, y1 - self.MIN_SIZE)
            if "s" in self._drag_mode:
                y1 = max(image_y, y0 + self.MIN_SIZE)
            self.bounds = [
                min(self.columns - self.MIN_SIZE, max(0, x0)),
                min(self.rows - self.MIN_SIZE, max(0, y0)),
                min(self.columns, max(self.MIN_SIZE, x1)),
                min(self.rows, max(self.MIN_SIZE, y1)),
            ]
        self._draw_selection()

    def _on_release(self, _event: tk.Event) -> None:
        self._drag_mode = None

    def _select_full(self) -> None:
        self.bounds = [0, 0, self.columns, self.rows]
        self._draw_selection()

    def _select_middle(self) -> None:
        row_start, row_stop, column_start, column_stop = crop_bounds_for_mode(
            self.values.shape, "xy"
        )
        self.bounds = [column_start, row_start, column_stop, row_stop]
        self._draw_selection()

    def _confirm(self) -> None:
        self._apply_numeric()
        x0, y0, x1, y1 = self.bounds
        self.result = validate_crop_bounds((y0, y1, x0, x1), self.values.shape)
        self.grab_release()
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()


class Infrared3DApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1480x900")
        self.minsize(1120, 700)
        self.configure(background="#e9edf2")

        self.raw_data: np.ndarray | None = None
        self.source_name = ""
        self.source_format = ""
        self.custom_crop_bounds: tuple[int, int, int, int] | None = None
        self.ax = None
        self._update_job: str | None = None
        self._view_update_job: str | None = None
        self._syncing_view_controls = False
        self._current_box_aspect: tuple[float, float, float] | None = None
        self._color_buttons: dict[str, tk.Button] = {}
        self.refinement_level = 0

        self.crop_var = tk.StringVar(value=DEFAULT_CROP_LABEL)
        self.crop_info_var = tk.StringVar(value="等待加载数据")
        self.pixel_size_var = tk.DoubleVar(value=0.01)
        self.immersion_var = tk.DoubleVar(value=90.0)
        self.absorption_var = tk.DoubleVar(value=80.0)

        self.above_color_var = tk.StringVar(value="#858B90")
        self.above_brightness_var = tk.DoubleVar(value=110.0)
        self.above_alpha_var = tk.DoubleVar(value=100.0)
        self.below_color_var = tk.StringVar(value="#5A7180")
        self.below_brightness_var = tk.DoubleVar(value=92.0)
        self.below_alpha_var = tk.DoubleVar(value=92.0)

        self.water_color_var = tk.StringVar(value="#258ED2")
        self.water_brightness_var = tk.DoubleVar(value=115.0)
        self.water_alpha_var = tk.DoubleVar(value=22.0)
        self.show_water_edges_var = tk.BooleanVar(value=True)

        self.gradient_enabled_var = tk.BooleanVar(value=True)
        self.gradient_strength_var = tk.DoubleVar(value=100.0)
        self.gradient_gamma_var = tk.DoubleVar(value=1.15)
        self.show_grid_var = tk.BooleanVar(value=False)
        self.grid_count_var = tk.IntVar(value=24)
        self.show_solid_walls_var = tk.BooleanVar(value=False)

        self.x_scale_var = tk.DoubleVar(value=1.0)
        self.y_scale_var = tk.DoubleVar(value=1.0)
        self.vertical_scale_var = tk.DoubleVar(value=1.0)
        self.view_preset_var = tk.StringVar(value=DEFAULT_VIEW_NAME)
        default_view = VIEW_PRESETS[DEFAULT_VIEW_NAME]
        self.projection_var = tk.StringVar(value=default_view.projection)
        self.rotation_x_var = tk.DoubleVar(value=default_view.rotation_x)
        self.rotation_y_var = tk.DoubleVar(value=default_view.rotation_y)
        self.rotation_z_var = tk.DoubleVar(value=default_view.rotation_z)
        self.view_zoom_var = tk.DoubleVar(value=default_view.zoom_percent)
        self.dpi_var = tk.IntVar(value=300)
        self.export_transparent_var = tk.BooleanVar(value=True)
        self.export_cropped_source_var = tk.BooleanVar(value=True)
        self.source_var = tk.StringVar(value="尚未加载数据")
        self.stats_var = tk.StringVar(value="读取 TXT/TIFF，或先载入内置示例查看效果。")
        self.status_var = tk.StringVar(value="就绪")
        self.refinement_info_var = tk.StringVar(value="细化次数：0")

        self._configure_style()
        self._build_ui()
        self._bind_parameter_updates()
        self.load_demo()

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        available = style.theme_names()
        style.theme_use("vista" if "vista" in available else "clam")
        style.configure("App.TFrame", background="#e9edf2")
        style.configure("Panel.TFrame", background="#f7f8fa")
        style.configure("Panel.TLabel", background="#f7f8fa", foreground="#26313c")
        style.configure("Panel.TCheckbutton", background="#f7f8fa", foreground="#26313c")
        style.configure(
            "Title.TLabel",
            background="#f7f8fa",
            foreground="#15202b",
            font=("Microsoft YaHei UI", 14, "bold"),
        )
        style.configure(
            "Section.TLabel",
            background="#f7f8fa",
            foreground="#26313c",
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.configure(
            "Hint.TLabel",
            background="#f7f8fa",
            foreground="#64717d",
            font=("Microsoft YaHei UI", 9),
        )
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 10, "bold"))

    def _build_ui(self) -> None:
        root = ttk.Frame(self, style="App.TFrame", padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        left_host = ttk.Frame(root, style="Panel.TFrame", width=430)
        left_host.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_host.grid_propagate(False)
        left_host.rowconfigure(0, weight=1)
        left_host.columnconfigure(0, weight=1)

        scroll_canvas = tk.Canvas(
            left_host,
            width=414,
            background="#f7f8fa",
            highlightthickness=0,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(left_host, orient="vertical", command=scroll_canvas.yview)
        controls = ttk.Frame(scroll_canvas, style="Panel.TFrame", padding=18)
        controls.columnconfigure(0, weight=1)
        window_id = scroll_canvas.create_window((0, 0), window=controls, anchor="nw")
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scroll_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        controls.bind(
            "<Configure>",
            lambda _event: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")),
        )
        scroll_canvas.bind(
            "<Configure>",
            lambda event: scroll_canvas.itemconfigure(window_id, width=event.width),
        )

        def enable_mousewheel(_event: tk.Event) -> None:
            self.bind_all(
                "<MouseWheel>",
                lambda event: scroll_canvas.yview_scroll(int(-event.delta / 120), "units"),
            )

        scroll_canvas.bind("<Enter>", enable_mousewheel)
        scroll_canvas.bind("<Leave>", lambda _event: self.unbind_all("<MouseWheel>"))

        row = 0
        ttk.Label(controls, text="红外 3D 示意图", style="Title.TLabel").grid(
            row=row, column=0, sticky="w"
        )
        row += 1
        ttk.Label(
            controls,
            text="分区材质 · 玻璃水体 · 温度渐变 · 可旋转导出",
            style="Hint.TLabel",
        ).grid(row=row, column=0, sticky="w", pady=(3, 16))

        row += 1
        button_row = ttk.Frame(controls, style="Panel.TFrame")
        button_row.grid(row=row, column=0, sticky="ew")
        button_row.columnconfigure((0, 1), weight=1)
        ttk.Button(
            button_row,
            text="读取 TXT / TIFF",
            command=self.open_file,
            style="Accent.TButton",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(button_row, text="载入示例", command=self.load_demo).grid(
            row=0, column=1, sticky="ew", padx=(5, 0)
        )

        row += 1
        ttk.Label(
            controls,
            textvariable=self.source_var,
            style="Hint.TLabel",
            wraplength=380,
            justify=tk.LEFT,
        ).grid(row=row, column=0, sticky="ew", pady=(8, 16))

        row += 1
        row = self._section_title(controls, row, "数据与截取")
        ttk.Label(controls, text="截取区域", style="Panel.TLabel").grid(
            row=row, column=0, sticky="w"
        )
        row += 1
        crop_actions = ttk.Frame(controls, style="Panel.TFrame")
        crop_actions.grid(row=row, column=0, sticky="ew", pady=(4, 5))
        crop_actions.columnconfigure(0, weight=1)
        ttk.Combobox(
            crop_actions,
            textvariable=self.crop_var,
            values=[label for label in CROP_LABELS if label != CUSTOM_CROP_LABEL],
            state="readonly",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(
            crop_actions,
            text="选择裁剪区域…",
            command=self.open_crop_selector,
        ).grid(row=0, column=1, sticky="ew")
        row += 1
        ttk.Label(
            controls,
            textvariable=self.crop_info_var,
            style="Hint.TLabel",
            wraplength=380,
        ).grid(row=row, column=0, sticky="w", pady=(0, 10))
        row += 1
        row = self._labeled_spinbox(
            controls, row, "单像素尺寸", self.pixel_size_var, 0.0001, 10.0, 0.001, "mm"
        )

        row = self._section_title(controls, row, "浸泡与吸收")
        row = self._labeled_spinbox(
            controls, row, "浸泡比例", self.immersion_var, 0.0, 100.0, 1.0, "%"
        )
        row = self._labeled_spinbox(
            controls,
            row,
            "水下红外吸收比例",
            self.absorption_var,
            0.0,
            100.0,
            1.0,
            "%",
        )
        ttk.Label(
            controls,
            text="水下相对水面的高度差 ×（1 − 吸收比例）",
            style="Hint.TLabel",
            wraplength=380,
        ).grid(row=row, column=0, sticky="w", pady=(0, 10))
        row += 1
        row = self._section_title(controls, row, "水面边界细化")
        ttk.Label(
            controls,
            textvariable=self.refinement_info_var,
            style="Panel.TLabel",
        ).grid(row=row, column=0, sticky="w", pady=(0, 7))
        row += 1
        refine_actions = ttk.Frame(controls, style="Panel.TFrame")
        refine_actions.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        refine_actions.columnconfigure((0, 1), weight=1)
        ttk.Button(refine_actions, text="细化一次", command=self.refine_once).grid(
            row=0, column=0, sticky="ew", padx=(0, 5)
        )
        ttk.Button(refine_actions, text="重置细化", command=self.reset_refinement).grid(
            row=0, column=1, sticky="ew", padx=(5, 0)
        )
        row += 1
        ttk.Label(
            controls,
            text="跨越水面的相邻点插入精确水面值；其余相邻点插入平均值。原始数据不会改变。",
            style="Hint.TLabel",
            wraplength=380,
        ).grid(row=row, column=0, sticky="w", pady=(0, 10))
        row += 1

        row = self._section_title(controls, row, "样品材质")
        ttk.Label(controls, text="水上区域", style="Section.TLabel").grid(
            row=row, column=0, sticky="w", pady=(0, 6)
        )
        row += 1
        row = self._color_row(controls, row, "基础颜色", self.above_color_var, "above")
        row = self._labeled_spinbox(
            controls, row, "亮度", self.above_brightness_var, 20.0, 220.0, 5.0, "%"
        )
        row = self._labeled_spinbox(
            controls, row, "不透明度", self.above_alpha_var, 5.0, 100.0, 1.0, "%"
        )

        ttk.Label(controls, text="水下区域", style="Section.TLabel").grid(
            row=row, column=0, sticky="w", pady=(5, 6)
        )
        row += 1
        row = self._color_row(controls, row, "基础颜色", self.below_color_var, "below")
        row = self._labeled_spinbox(
            controls, row, "亮度", self.below_brightness_var, 20.0, 220.0, 5.0, "%"
        )
        row = self._labeled_spinbox(
            controls, row, "不透明度", self.below_alpha_var, 5.0, 100.0, 1.0, "%"
        )
        ttk.Checkbutton(
            controls,
            text="显示样品网格",
            variable=self.show_grid_var,
            style="Panel.TCheckbutton",
        ).grid(row=row, column=0, sticky="w", pady=(1, 7))
        row += 1
        row = self._labeled_spinbox(
            controls, row, "网格线数量", self.grid_count_var, 4, 60, 1, "条/方向"
        )
        ttk.Checkbutton(
            controls,
            text="显示固体侧壁（裁切面）",
            variable=self.show_solid_walls_var,
            style="Panel.TCheckbutton",
        ).grid(row=row, column=0, sticky="w", pady=(0, 9))
        row += 1

        row = self._section_title(controls, row, "三维玻璃水体")
        row = self._color_row(controls, row, "水体颜色", self.water_color_var, "water")
        row = self._labeled_spinbox(
            controls, row, "水体亮度", self.water_brightness_var, 20.0, 220.0, 5.0, "%"
        )
        row = self._labeled_spinbox(
            controls, row, "水体不透明度", self.water_alpha_var, 2.0, 90.0, 1.0, "%"
        )
        ttk.Checkbutton(
            controls,
            text="显示玻璃体轮廓高光",
            variable=self.show_water_edges_var,
            style="Panel.TCheckbutton",
        ).grid(row=row, column=0, sticky="w", pady=(1, 9))
        row += 1

        row = self._section_title(controls, row, "温度渐变")
        ttk.Checkbutton(
            controls,
            text="启用顶部白化渐变",
            variable=self.gradient_enabled_var,
            style="Panel.TCheckbutton",
        ).grid(row=row, column=0, sticky="w", pady=(0, 8))
        row += 1
        row = self._labeled_spinbox(
            controls,
            row,
            "白化强度",
            self.gradient_strength_var,
            0.0,
            100.0,
            5.0,
            "%",
        )
        row = self._labeled_spinbox(
            controls,
            row,
            "渐变曲线",
            self.gradient_gamma_var,
            0.2,
            5.0,
            0.05,
            "γ",
        )
        ttk.Label(
            controls,
            text="γ 越小，较大范围会变亮；γ 越大，白色越集中在峰顶。",
            style="Hint.TLabel",
            wraplength=380,
        ).grid(row=row, column=0, sticky="w", pady=(0, 10))
        row += 1

        row = self._section_title(controls, row, "整体显示比例")
        row = self._labeled_spinbox(
            controls,
            row,
            "X 显示比例",
            self.x_scale_var,
            0.1,
            10.0,
            0.1,
            "×",
        )
        row = self._labeled_spinbox(
            controls,
            row,
            "Y 显示比例",
            self.y_scale_var,
            0.1,
            10.0,
            0.1,
            "×",
        )
        row = self._labeled_spinbox(
            controls,
            row,
            "Z 显示比例",
            self.vertical_scale_var,
            0.01,
            20.0,
            0.01,
            "×",
        )
        aspect_actions = ttk.Frame(controls, style="Panel.TFrame")
        aspect_actions.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        aspect_actions.columnconfigure((0, 1), weight=1)
        ttk.Button(
            aspect_actions, text="扁平长方体预设", command=self.set_flat_aspect
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(aspect_actions, text="重置整体比例", command=self.reset_aspect).grid(
            row=0, column=1, sticky="ew", padx=(5, 0)
        )
        row += 1

        row = self._section_title(controls, row, "定量视角")
        view_wrapper = ttk.Frame(controls, style="Panel.TFrame")
        view_wrapper.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        view_wrapper.columnconfigure(1, weight=1)
        ttk.Label(view_wrapper, text="视角预设", style="Panel.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 12)
        )
        view_box = ttk.Combobox(
            view_wrapper,
            textvariable=self.view_preset_var,
            values=(*VIEW_PRESETS, CUSTOM_VIEW_NAME),
            state="readonly",
            width=22,
        )
        view_box.grid(row=0, column=1, sticky="ew")
        view_box.bind("<<ComboboxSelected>>", self.apply_view_preset)
        row += 1

        projection_wrapper = ttk.Frame(controls, style="Panel.TFrame")
        projection_wrapper.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        projection_wrapper.columnconfigure(1, weight=1)
        ttk.Label(projection_wrapper, text="投影模式", style="Panel.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 12)
        )
        ttk.Combobox(
            projection_wrapper,
            textvariable=self.projection_var,
            values=tuple(PROJECTION_MODES),
            state="readonly",
            width=22,
        ).grid(row=0, column=1, sticky="ew")
        row += 1

        row = self._labeled_spinbox(
            controls, row, "样品 X 旋转", self.rotation_x_var, -180.0, 180.0, 1.0, "°"
        )
        row = self._labeled_spinbox(
            controls, row, "样品 Y 旋转", self.rotation_y_var, -180.0, 180.0, 1.0, "°"
        )
        row = self._labeled_spinbox(
            controls, row, "样品 Z 旋转", self.rotation_z_var, -180.0, 180.0, 1.0, "°"
        )
        row = self._labeled_spinbox(
            controls, row, "视图缩放比例", self.view_zoom_var, 20.0, 300.0, 5.0, "%"
        )
        ttk.Label(
            controls,
            text="0° / 0° / 0° 为正视；X 为俯仰、Y 为水平旋转、Z 为画面滚转。拖动后数值会自动同步。",
            style="Hint.TLabel",
            wraplength=380,
        ).grid(row=row, column=0, sticky="w", pady=(0, 9))
        row += 1

        cube_views = ttk.Frame(controls, style="Panel.TFrame")
        cube_views.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        cube_views.columnconfigure((0, 1, 2), weight=1)
        for column, (label, preset) in enumerate(
            (
                ("面视角", "立方体 · 面（正视）"),
                ("棱视角", "立方体 · 棱（双面）"),
                ("角视角", "立方体 · 角（等轴测）"),
            )
        ):
            ttk.Button(
                cube_views,
                text=label,
                command=lambda name=preset: self.set_view_preset(name),
            ).grid(
                row=0,
                column=column,
                sticky="ew",
                padx=(0 if column == 0 else 4, 0 if column == 2 else 4),
            )
        row += 1

        row = self._section_title(controls, row, "导出")
        row = self._labeled_spinbox(
            controls, row, "导出分辨率", self.dpi_var, 72, 1200, 10, "DPI"
        )
        ttk.Checkbutton(
            controls,
            text="导出透明背景（PNG / TIFF）",
            variable=self.export_transparent_var,
            style="Panel.TCheckbutton",
        ).grid(row=row, column=0, sticky="w", pady=(0, 10))
        row += 1
        ttk.Checkbutton(
            controls,
            text="同时导出当前裁剪范围的源数据副本",
            variable=self.export_cropped_source_var,
            style="Panel.TCheckbutton",
        ).grid(row=row, column=0, sticky="w", pady=(0, 10))
        row += 1

        actions = ttk.Frame(controls, style="Panel.TFrame")
        actions.grid(row=row, column=0, sticky="ew", pady=(4, 0))
        actions.columnconfigure((0, 1), weight=1)
        ttk.Button(actions, text="更新预览", command=self.update_plot).grid(
            row=0, column=0, sticky="ew", padx=(0, 5)
        )
        ttk.Button(actions, text="重置视角", command=self.reset_view).grid(
            row=0, column=1, sticky="ew", padx=(5, 0)
        )
        row += 1
        ttk.Button(
            controls,
            text="导出 PNG / TIFF",
            command=self.export_image,
            style="Accent.TButton",
        ).grid(row=row, column=0, sticky="ew", pady=(10, 14))

        row += 1
        ttk.Separator(controls).grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        ttk.Label(
            controls,
            textvariable=self.stats_var,
            style="Hint.TLabel",
            wraplength=380,
            justify=tk.LEFT,
        ).grid(row=row, column=0, sticky="ew")

        right = ttk.Frame(root, style="Panel.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        top = ttk.Frame(right, style="Panel.TFrame", padding=(12, 9))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        ttk.Label(
            top,
            text="左键拖动旋转并同步角度 · 滚轮缩放 · 数字视角可精确复现",
            style="Hint.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(top, textvariable=self.status_var, style="Hint.TLabel").grid(
            row=0, column=1, sticky="e"
        )

        figure_host = tk.Frame(right, background="#f4f6f8")
        figure_host.grid(row=1, column=0, sticky="nsew")
        figure_host.rowconfigure(0, weight=1)
        figure_host.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(9, 7), dpi=100, facecolor="#f4f6f8")
        self.canvas = FigureCanvasTkAgg(self.figure, master=figure_host)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_host = ttk.Frame(right, style="Panel.TFrame")
        toolbar_host.grid(row=2, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_host, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.LEFT, padx=6, pady=3)
        self.canvas.mpl_connect("button_release_event", self._on_view_interaction_end)
        self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)

    def _section_title(self, parent: ttk.Frame, row: int, text: str) -> int:
        ttk.Separator(parent).grid(row=row, column=0, sticky="ew", pady=(8, 12))
        row += 1
        ttk.Label(parent, text=text, style="Section.TLabel").grid(
            row=row, column=0, sticky="w", pady=(0, 9)
        )
        return row + 1

    def _labeled_spinbox(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.Variable,
        start: float,
        stop: float,
        increment: float,
        unit: str,
    ) -> int:
        wrapper = ttk.Frame(parent, style="Panel.TFrame")
        wrapper.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        wrapper.columnconfigure(0, weight=1)
        ttk.Label(wrapper, text=label, style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        spinbox = ttk.Spinbox(
            wrapper,
            textvariable=variable,
            from_=start,
            to=stop,
            increment=increment,
            width=11,
        )
        spinbox.grid(row=0, column=1, sticky="e", padx=(12, 7))
        ttk.Label(wrapper, text=unit, style="Hint.TLabel").grid(row=0, column=2, sticky="w")
        return row + 1

    def _color_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        key: str,
    ) -> int:
        wrapper = ttk.Frame(parent, style="Panel.TFrame")
        wrapper.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        wrapper.columnconfigure(0, weight=1)
        ttk.Label(wrapper, text=label, style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        button = tk.Button(
            wrapper,
            text=variable.get(),
            width=11,
            relief=tk.GROOVE,
            borderwidth=1,
            command=lambda: self._choose_color(variable, key, label),
        )
        button.grid(row=0, column=1, sticky="e")
        self._color_buttons[key] = button
        self._refresh_color_button(key, variable.get())
        return row + 1

    def _choose_color(self, variable: tk.StringVar, key: str, label: str) -> None:
        _rgb, selected = colorchooser.askcolor(
            color=variable.get(), title=f"选择{label}", parent=self
        )
        if selected:
            variable.set(selected.upper())
            self._refresh_color_button(key, selected)

    def _refresh_color_button(self, key: str, color: str) -> None:
        button = self._color_buttons.get(key)
        if button is None:
            return
        rgb = np.asarray(to_rgb(color))
        luminance = float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
        foreground = "#111111" if luminance > 0.55 else "#FFFFFF"
        button.configure(
            text=color.upper(),
            background=color,
            activebackground=color,
            foreground=foreground,
            activeforeground=foreground,
        )

    def _bind_parameter_updates(self) -> None:
        variables = (
            self.pixel_size_var,
            self.immersion_var,
            self.absorption_var,
            self.above_color_var,
            self.above_brightness_var,
            self.above_alpha_var,
            self.below_color_var,
            self.below_brightness_var,
            self.below_alpha_var,
            self.water_color_var,
            self.water_brightness_var,
            self.water_alpha_var,
            self.show_water_edges_var,
            self.gradient_enabled_var,
            self.gradient_strength_var,
            self.gradient_gamma_var,
            self.show_grid_var,
            self.grid_count_var,
            self.show_solid_walls_var,
            self.x_scale_var,
            self.y_scale_var,
            self.vertical_scale_var,
        )
        for variable in variables:
            variable.trace_add("write", self._schedule_update)
        self.crop_var.trace_add("write", self._on_crop_mode_changed)
        for variable in (
            self.projection_var,
            self.rotation_x_var,
            self.rotation_y_var,
            self.rotation_z_var,
            self.view_zoom_var,
        ):
            variable.trace_add("write", self._schedule_view_update)

    def _schedule_update(self, *_args: object) -> None:
        if self.raw_data is None:
            return
        if self._update_job is not None:
            self.after_cancel(self._update_job)
        self._update_job = self.after(350, self.update_plot)

    def _on_crop_mode_changed(self, *_args: object) -> None:
        self._refresh_crop_info()
        self._schedule_update()

    def _current_crop_bounds(
        self, crop_mode: str | None = None
    ) -> tuple[int, int, int, int]:
        if self.raw_data is None:
            raise ValueError("请先读取温度数据。")
        mode = crop_mode or CROP_LABELS[self.crop_var.get()]
        if mode == "custom":
            if self.custom_crop_bounds is None:
                raise ValueError("请先在裁剪选择窗口中确定自定义区域。")
            return validate_crop_bounds(self.custom_crop_bounds, self.raw_data.shape)
        return crop_bounds_for_mode(self.raw_data.shape, mode)

    def _crop_current_source(self, crop_mode: str | None = None) -> np.ndarray:
        if self.raw_data is None:
            raise ValueError("请先读取温度数据。")
        mode = crop_mode or CROP_LABELS[self.crop_var.get()]
        return crop_temperature(
            self.raw_data,
            mode,
            self.custom_crop_bounds if mode == "custom" else None,
        )

    def _refresh_crop_info(self) -> None:
        if self.raw_data is None:
            self.crop_info_var.set("等待加载数据")
            return
        try:
            row_start, row_stop, column_start, column_stop = self._current_crop_bounds()
        except (ValueError, KeyError):
            self.crop_info_var.set("尚未选择有效的自定义区域")
            return
        self.crop_info_var.set(
            f"X={column_start}, Y={row_start}, "
            f"尺寸={column_stop - column_start} × {row_stop - row_start} px"
        )

    def open_crop_selector(self) -> None:
        if self.raw_data is None:
            messagebox.showinfo("没有数据", "请先读取 TXT 或 TIFF。", parent=self)
            return
        try:
            initial_bounds = self._current_crop_bounds()
        except (ValueError, KeyError):
            initial_bounds = crop_bounds_for_mode(self.raw_data.shape, "xy")
        dialog = CropSelectionDialog(self, self.raw_data, initial_bounds)
        self.wait_window(dialog)
        if dialog.result is None:
            return
        self.custom_crop_bounds = dialog.result
        self.refinement_level = 0
        self.refinement_info_var.set("细化次数：0")
        self.crop_var.set(CUSTOM_CROP_LABEL)
        self._refresh_crop_info()
        self.status_var.set("已应用自定义裁剪区域")

    def _schedule_view_update(self, *_args: object) -> None:
        if self._syncing_view_controls or self.ax is None:
            return
        self.view_preset_var.set(CUSTOM_VIEW_NAME)
        if self._view_update_job is not None:
            self.after_cancel(self._view_update_job)
        self._view_update_job = self.after(120, self._run_scheduled_view_update)

    def refine_once(self) -> None:
        if self.raw_data is None:
            return
        new_level = self.refinement_level + 1
        if new_level > MAX_REFINEMENT_LEVEL:
            messagebox.showinfo(
                "已达到细化上限",
                f"最多允许细化 {MAX_REFINEMENT_LEVEL} 次。",
                parent=self,
            )
            return
        try:
            cropped = self._crop_current_source()
        except (ValueError, KeyError) as exc:
            messagebox.showerror("无法细化", str(exc), parent=self)
            return
        factor = 2**new_level
        rows = (cropped.shape[0] - 1) * factor + 1
        columns = (cropped.shape[1] - 1) * factor + 1
        if rows * columns > MAX_REFINED_POINTS:
            messagebox.showwarning(
                "细化点数过多",
                f"下一次细化将产生 {columns} × {rows} 个点，超过安全上限 "
                f"{MAX_REFINED_POINTS:,}。\n\n可先缩小截取区域，或保持当前细化级别。",
                parent=self,
            )
            return
        self.refinement_level = new_level
        self.refinement_info_var.set(
            f"细化次数：{new_level}（绘图网格 {columns} × {rows}）"
        )
        self.status_var.set(f"正在执行第 {new_level} 次细化…")
        self.update_plot()

    def reset_refinement(self) -> None:
        self.refinement_level = 0
        self.refinement_info_var.set("细化次数：0")
        self.status_var.set("细化已重置")
        self.update_plot()

    def set_flat_aspect(self) -> None:
        self.x_scale_var.set(3.0)
        self.y_scale_var.set(1.0)
        self.vertical_scale_var.set(0.12)
        self.status_var.set("已应用扁平长方体比例")

    def reset_aspect(self) -> None:
        self.x_scale_var.set(1.0)
        self.y_scale_var.set(1.0)
        self.vertical_scale_var.set(1.0)
        self.status_var.set("整体比例已重置")

    def open_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择红外温度 TXT 或 TIFF",
            filetypes=[
                ("支持的数据", "*.txt *.csv *.dat *.tif *.tiff"),
                ("TIFF 图像", "*.tif *.tiff"),
                ("温度文本", "*.txt *.csv *.dat"),
                ("所有文件", "*.*"),
            ],
        )
        if not selected:
            return
        self.load_path(selected)

    def load_path(self, selected: str | Path) -> bool:
        selected = str(selected)
        try:
            loaded = load_temperature_file(selected)
        except Exception as exc:
            messagebox.showerror("无法读取数据", str(exc), parent=self)
            self.status_var.set("读取失败")
            return False

        self.raw_data = loaded.values
        self.custom_crop_bounds = None
        if self.crop_var.get() == CUSTOM_CROP_LABEL:
            self.crop_var.set(DEFAULT_CROP_LABEL)
        self.refinement_level = 0
        self.refinement_info_var.set("细化次数：0")
        self.source_name = str(Path(selected))
        self.source_format = loaded.source_format
        self.title(f"{APP_TITLE} · {Path(selected).name}")
        self.source_var.set(
            f"{Path(selected).name}\n{loaded.values.shape[1]} × {loaded.values.shape[0]} px · {loaded.source_format}"
        )
        self._refresh_crop_info()
        self.status_var.set("已读取数据")
        self.update_plot(reset_camera=True)
        return True

    def load_demo(self) -> None:
        self.raw_data = make_demo_data()
        self.custom_crop_bounds = None
        if self.crop_var.get() == CUSTOM_CROP_LABEL:
            self.crop_var.set(DEFAULT_CROP_LABEL)
        self.refinement_level = 0
        self.refinement_info_var.set("细化次数：0")
        self.source_name = "内置示例"
        self.source_format = "二维温度矩阵"
        self.title(APP_TITLE)
        self.source_var.set(
            f"内置示例\n{self.raw_data.shape[1]} × {self.raw_data.shape[0]} px · 二维温度矩阵"
        )
        self._refresh_crop_info()
        self.status_var.set("正在生成预览…")
        self.after_idle(lambda: self.update_plot(reset_camera=True))

    def _read_settings(self) -> RenderSettings:
        settings = RenderSettings(
            crop_mode=CROP_LABELS[self.crop_var.get()],
            pixel_size=float(self.pixel_size_var.get()),
            immersion=float(self.immersion_var.get()) / 100.0,
            absorption=float(self.absorption_var.get()) / 100.0,
            x_scale=float(self.x_scale_var.get()),
            y_scale=float(self.y_scale_var.get()),
            vertical_scale=float(self.vertical_scale_var.get()),
            above_color=to_rgb(self.above_color_var.get()),
            above_brightness=float(self.above_brightness_var.get()) / 100.0,
            above_alpha=float(self.above_alpha_var.get()) / 100.0,
            below_color=to_rgb(self.below_color_var.get()),
            below_brightness=float(self.below_brightness_var.get()) / 100.0,
            below_alpha=float(self.below_alpha_var.get()) / 100.0,
            water_color=to_rgb(self.water_color_var.get()),
            water_brightness=float(self.water_brightness_var.get()) / 100.0,
            water_alpha=float(self.water_alpha_var.get()) / 100.0,
            gradient_enabled=bool(self.gradient_enabled_var.get()),
            gradient_strength=float(self.gradient_strength_var.get()) / 100.0,
            gradient_gamma=float(self.gradient_gamma_var.get()),
            show_grid=bool(self.show_grid_var.get()),
            grid_count=int(self.grid_count_var.get()),
            show_solid_walls=bool(self.show_solid_walls_var.get()),
            show_water_edges=bool(self.show_water_edges_var.get()),
        )
        if settings.pixel_size <= 0:
            raise ValueError("单像素尺寸必须大于 0。")
        if not 0 <= settings.immersion <= 1:
            raise ValueError("浸泡比例必须在 0–100% 之间。")
        if not 0 <= settings.absorption <= 1:
            raise ValueError("吸收比例必须在 0–100% 之间。")
        if not 0.02 <= settings.water_alpha <= 0.90:
            raise ValueError("水体不透明度必须在 2–90% 之间。")
        if not 0.05 <= settings.above_alpha <= 1 or not 0.05 <= settings.below_alpha <= 1:
            raise ValueError("样品不透明度必须在 5–100% 之间。")
        if not 0.2 <= settings.gradient_gamma <= 5:
            raise ValueError("渐变曲线必须在 0.2–5 之间。")
        if not 0.1 <= settings.x_scale <= 10 or not 0.1 <= settings.y_scale <= 10:
            raise ValueError("X/Y 显示比例必须在 0.1–10 之间。")
        if not 0.01 <= settings.vertical_scale <= 20:
            raise ValueError("Z 显示比例必须在 0.01–20 之间。")
        if not 4 <= settings.grid_count <= 60:
            raise ValueError("网格线数量必须在 4–60 之间。")
        return settings

    def update_plot(self, reset_camera: bool = False) -> None:
        self._update_job = None
        if self.raw_data is None:
            return

        if reset_camera:
            self._set_view_controls_from_preset(DEFAULT_VIEW_NAME)

        try:
            settings = self._read_settings()
            projection, camera, view_zoom = self._read_numeric_view()
            cropped = self._crop_current_source(settings.crop_mode)
            factor = 2**self.refinement_level
            projected_rows = (cropped.shape[0] - 1) * factor + 1
            projected_columns = (cropped.shape[1] - 1) * factor + 1
            if projected_rows * projected_columns > MAX_REFINED_POINTS:
                raise ValueError(
                    "当前截取区域与细化次数会产生过多网格点；请重置一次细化或缩小截取区域。"
                )
            base_result = apply_immersion_absorption(
                cropped, settings.immersion, settings.absorption
            )
            plot_data = refine_temperature_grid(
                cropped,
                base_result.waterline_temperature,
                levels=self.refinement_level,
            )
            result = apply_immersion_absorption(
                plot_data, settings.immersion, settings.absorption
            )
        except (ValueError, tk.TclError) as exc:
            self.status_var.set(str(exc))
            return

        display_rows = plot_data.shape[0] if self.refinement_level > 0 else 160
        display_columns = plot_data.shape[1] if self.refinement_level > 0 else 160
        z_small, row_indices, column_indices = downsample_grid(
            result.display_height, display_rows, display_columns
        )
        raw_small = plot_data[np.ix_(row_indices, column_indices)]
        refined_pixel_size = settings.pixel_size / (2**self.refinement_level)
        x = column_indices.astype(float) * refined_pixel_size
        y = row_indices.astype(float) * refined_pixel_size
        x -= 0.5 * (x[0] + x[-1])
        y -= 0.5 * (y[0] + y[-1])
        xx, yy = np.meshgrid(x, y)

        self.figure.clear()
        # Keep every face and grid segment in a single collection.  Matplotlib
        # can then depth-sort the actual geometry instead of painting whole
        # materials in a fixed order.
        self.ax = self.figure.add_subplot(111, projection="3d", computed_zorder=False)
        self.ax.set_facecolor("#f4f6f8")
        scene = _SceneMesh()
        sample_base = self._draw_sample(
            xx,
            yy,
            z_small,
            raw_small,
            result.waterline_temperature,
            settings,
            scene,
        )
        water_bottom, water_top = self._draw_glass_water(
            xx,
            yy,
            result.waterline_temperature,
            plot_data,
            raw_small,
            sample_base,
            settings,
            scene,
        )
        self.ax.add_collection3d(scene.to_collection())

        x_span = max(float(np.ptp(x)), refined_pixel_size)
        y_span = max(float(np.ptp(y)), refined_pixel_size)
        z_min = min(float(np.min(z_small)), sample_base, water_bottom)
        z_max = max(float(np.max(z_small)), water_top)
        z_span = max(z_max - z_min, 1e-6)
        display_x = x_span * settings.x_scale
        display_y = y_span * settings.y_scale
        z_box = max(
            z_span * settings.vertical_scale,
            max(display_x, display_y) * 0.002,
        )
        self.ax.set_xlim(float(np.min(x)), float(np.max(x)))
        self.ax.set_ylim(float(np.min(y)), float(np.max(y)))
        self.ax.set_zlim(z_min, z_max)
        self._current_box_aspect = (display_x, display_y, z_box)
        self.ax.set_box_aspect(self._current_box_aspect, zoom=view_zoom)
        self.ax.set_proj_type(projection)
        self.ax.set_axis_off()
        self.ax.view_init(elev=camera[0], azim=camera[1], roll=camera[2])
        self.ax.margins(0)
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas.draw_idle()

        submerged_percent = 100.0 * float(np.mean(result.submerged_mask))
        self.refinement_info_var.set(
            f"细化次数：{self.refinement_level}（绘图网格 "
            f"{plot_data.shape[1]} × {plot_data.shape[0]}）"
        )
        self.stats_var.set(
            f"原始：{self.raw_data.shape[1]} × {self.raw_data.shape[0]} px\n"
            f"当前区域：{cropped.shape[1]} × {cropped.shape[0]} px\n"
            f"绘图网格：{plot_data.shape[1]} × {plot_data.shape[0]}\n"
            f"数据范围：{np.min(cropped):.4g}–{np.max(cropped):.4g}\n"
            f"水面阈值：{result.waterline_temperature:.4g}\n"
            f"水下像素：{submerged_percent:.1f}%"
        )
        self.status_var.set("预览已更新")

    def _draw_sample(
        self,
        xx: np.ndarray,
        yy: np.ndarray,
        z: np.ndarray,
        raw: np.ndarray,
        waterline: float,
        settings: RenderSettings,
        scene: _SceneMesh,
    ) -> float:
        normalized_height = z - float(np.min(z))
        height_span = float(np.ptp(normalized_height))
        if height_span > 0:
            normalized_height /= height_span
        light = LightSource(azdeg=310, altdeg=42)
        illumination = light.hillshade(
            normalized_height, vert_exag=1.25, dx=1.0, dy=1.0
        )

        thermal = raw - float(np.min(raw))
        thermal_span = float(np.ptp(thermal))
        if thermal_span > 0:
            thermal /= thermal_span
        submerged = raw < waterline

        above_base = _adjust_rgb(settings.above_color, settings.above_brightness)
        below_base = _adjust_rgb(settings.below_color, settings.below_brightness)
        base_rgb = np.where(submerged[..., None], below_base, above_base)
        light_factor = 0.72 + 0.42 * illumination
        rgb = np.clip(base_rgb * light_factor[..., None], 0.0, 1.0)

        if settings.gradient_enabled:
            white_mix = settings.gradient_strength * np.power(
                np.clip(thermal, 0.0, 1.0), settings.gradient_gamma
            )
            rgb = rgb * (1.0 - white_mix[..., None]) + white_mix[..., None]

        specular = (np.clip(illumination, 0, 1) ** 10 * 0.08)[..., None]
        rgb = np.clip(rgb + specular, 0.0, 1.0)
        alpha = np.where(submerged, settings.below_alpha, settings.above_alpha)
        rgba = np.concatenate([rgb, alpha[..., None]], axis=2)

        scene.add_surface(xx, yy, z, rgba)

        if settings.show_grid:
            row_stride = max(1, int(np.ceil(z.shape[0] / settings.grid_count)))
            column_stride = max(1, int(np.ceil(z.shape[1] / settings.grid_count)))
            grid_z = z + max(float(np.ptp(z)), 0.01) * 0.001
            row_positions = list(range(0, z.shape[0], row_stride))
            column_positions = list(range(0, z.shape[1], column_stride))
            if row_positions[-1] != z.shape[0] - 1:
                row_positions.append(z.shape[0] - 1)
            if column_positions[-1] != z.shape[1] - 1:
                column_positions.append(z.shape[1] - 1)
            grid_color = (0.08, 0.10, 0.12, 0.38)
            for row in row_positions:
                scene.add_polyline(xx[row], yy[row], grid_z[row], grid_color)
            for column in column_positions:
                scene.add_polyline(
                    xx[:, column], yy[:, column], grid_z[:, column], grid_color
                )

        z_range = max(float(np.ptp(z)), 0.01)
        base_z = float(np.min(z)) - 0.12 * z_range
        # Omit cut walls entirely by default. An alpha-zero Poly3DCollection
        # would still participate in Matplotlib's 3-D depth sorting and could
        # incorrectly cover the surface when a peak is clipped at an edge.
        if not settings.show_solid_walls:
            return base_z

        walls: list[list[tuple[float, float, float]]] = []
        wall_colors: list[tuple[float, float, float, float]] = []
        boundaries = [
            (xx[0, :], yy[0, :], z[0, :], raw[0, :]),
            (xx[-1, :], yy[-1, :], z[-1, :], raw[-1, :]),
            (xx[:, 0], yy[:, 0], z[:, 0], raw[:, 0]),
            (xx[:, -1], yy[:, -1], z[:, -1], raw[:, -1]),
        ]
        raw_min = float(np.min(raw))
        raw_span = max(float(np.ptp(raw)), 1e-12)
        for edge_x, edge_y, edge_z, edge_raw in boundaries:
            for index in range(len(edge_x) - 1):
                walls.append(
                    [
                        (edge_x[index], edge_y[index], edge_z[index]),
                        (edge_x[index + 1], edge_y[index + 1], edge_z[index + 1]),
                        (edge_x[index + 1], edge_y[index + 1], base_z),
                        (edge_x[index], edge_y[index], base_z),
                    ]
                )
                mean_raw = 0.5 * (edge_raw[index] + edge_raw[index + 1])
                is_below = mean_raw < waterline
                wall_rgb = below_base if is_below else above_base
                wall_rgb = np.clip(wall_rgb * 0.72, 0.0, 1.0)
                if settings.gradient_enabled:
                    thermal_level = np.clip((mean_raw - raw_min) / raw_span, 0.0, 1.0)
                    mix = 0.28 * settings.gradient_strength * thermal_level**settings.gradient_gamma
                    wall_rgb = wall_rgb * (1.0 - mix) + mix
                wall_alpha = settings.below_alpha if is_below else settings.above_alpha
                wall_colors.append((*wall_rgb, wall_alpha))

        scene.add_faces(
            walls,
            np.asarray(wall_colors),
            (0.15, 0.17, 0.19, 0.12),
        )
        bottom_rgb = np.clip(below_base * 0.62, 0.0, 1.0)
        scene.add_faces(
            [[
                (xx[0, 0], yy[0, 0], base_z),
                (xx[0, -1], yy[0, -1], base_z),
                (xx[-1, -1], yy[-1, -1], base_z),
                (xx[-1, 0], yy[-1, 0], base_z),
            ]],
            (*bottom_rgb, settings.below_alpha),
        )
        return base_z

    def _draw_glass_water(
        self,
        xx: np.ndarray,
        yy: np.ndarray,
        waterline: float,
        temperature: np.ndarray,
        raw_surface: np.ndarray,
        sample_base: float,
        settings: RenderSettings,
        scene: _SceneMesh,
    ) -> tuple[float, float]:
        # The water footprint is intentionally identical to the solid footprint.
        # It must not extend around or below the sample boundary.
        x_min, x_max = float(np.min(xx)), float(np.max(xx))
        y_min, y_max = float(np.min(yy)), float(np.max(yy))
        water_x = xx[0, :]
        water_y = yy[:, 0]
        water_xx, water_yy = xx, yy

        normalized_x = (water_xx - water_xx.min()) / max(float(np.ptp(water_xx)), 1e-9)
        normalized_y = (water_yy - water_yy.min()) / max(float(np.ptp(water_yy)), 1e-9)
        temperature_span = max(float(np.ptp(temperature)), 0.05)
        ripple_amplitude = 0.006 * temperature_span
        ripples = ripple_amplitude * (
            0.58 * np.sin(3.4 * np.pi * normalized_x + 1.1 * normalized_y)
            + 0.42 * np.cos(4.1 * np.pi * normalized_y - 0.8 * normalized_x)
        )
        water_z = waterline + ripples
        # Cut the horizontal water surface away wherever the sample protrudes
        # above the waterline. This avoids tinting the dry peaks blue.
        water_z_visible = water_z.copy()
        water_z_visible[raw_surface >= waterline] = np.nan

        base_rgb = _adjust_rgb(settings.water_color, settings.water_brightness)
        water_rgba = np.empty((*water_z.shape, 4), dtype=float)
        # Use one material color and one opacity for the top, sides and bottom.
        # The previous per-face multipliers (0.78x / 1.22x / 0.16x), combined
        # with different white highlights, made red water look grey on top but
        # saturated red at the crop boundary.
        water_rgba[..., :3] = base_rgb
        water_rgba[..., 3] = settings.water_alpha
        if np.any(np.isfinite(water_z_visible)):
            scene.add_surface(water_xx, water_yy, water_z_visible, water_rgba)

        water_bottom = sample_base
        side_faces: list[list[tuple[float, float, float]]] = []
        side_colors: list[tuple[float, float, float, float]] = []
        perimeter = [
            (water_xx[0, :], water_yy[0, :], water_z[0, :]),
            (water_xx[-1, :], water_yy[-1, :], water_z[-1, :]),
            (water_xx[:, 0], water_yy[:, 0], water_z[:, 0]),
            (water_xx[:, -1], water_yy[:, -1], water_z[:, -1]),
        ]
        for edge_x, edge_y, edge_z in perimeter:
            for index in range(len(edge_x) - 1):
                side_faces.append(
                    [
                        (edge_x[index], edge_y[index], edge_z[index]),
                        (edge_x[index + 1], edge_y[index + 1], edge_z[index + 1]),
                        (edge_x[index + 1], edge_y[index + 1], water_bottom),
                        (edge_x[index], edge_y[index], water_bottom),
                    ]
                )
                side_colors.append((*base_rgb, settings.water_alpha))
        scene.add_faces(side_faces, np.asarray(side_colors))

        # Do not add a horizontal face at ``water_bottom``.  That face would be
        # completely hidden below the sample in a real scene, but it used to be
        # one footprint-sized translucent quad.  Matplotlib sorts a polygon by
        # one average depth, so at near-horizontal views half of that enormous
        # rear quad could be painted over a foreground dry peak.  The water top
        # and its four perimeter walls already provide the intended glass-like
        # volume without introducing an invisible layer that can leak through.

        if settings.show_water_edges:
            edge_rgb = np.clip(base_rgb * 0.85 + 0.15, 0.0, 1.0)
            edge_color = (*edge_rgb, settings.water_alpha)
            for edge_x, edge_y, edge_z in perimeter:
                scene.add_polyline(edge_x, edge_y, edge_z, edge_color)
            corners = [
                (water_x[0], water_y[0], water_z[0, 0]),
                (water_x[-1], water_y[0], water_z[0, -1]),
                (water_x[-1], water_y[-1], water_z[-1, -1]),
                (water_x[0], water_y[-1], water_z[-1, 0]),
            ]
            for corner_x, corner_y, corner_z in corners:
                scene.add_polyline(
                    [corner_x, corner_x],
                    [corner_y, corner_y],
                    [water_bottom, corner_z],
                    edge_color,
                )

        visible_top = (
            float(np.nanmax(water_z_visible))
            if np.any(np.isfinite(water_z_visible))
            else waterline
        )
        return water_bottom, visible_top

    def set_view_preset(self, name: str) -> None:
        if name not in VIEW_PRESETS:
            return
        self.view_preset_var.set(name)
        self.apply_view_preset()

    def _read_numeric_view(
        self,
    ) -> tuple[str, tuple[float, float, float], float]:
        projection_label = self.projection_var.get()
        if projection_label not in PROJECTION_MODES:
            raise ValueError("请选择透视投影或正交投影。")
        rotation_x = float(self.rotation_x_var.get())
        rotation_y = float(self.rotation_y_var.get())
        rotation_z = float(self.rotation_z_var.get())
        if not all(
            -180.0 <= angle <= 180.0
            for angle in (rotation_x, rotation_y, rotation_z)
        ):
            raise ValueError("样品 X/Y/Z 旋转角必须在 −180°–180° 之间。")
        zoom_percent = float(self.view_zoom_var.get())
        if not 20.0 <= zoom_percent <= 300.0:
            raise ValueError("视图缩放比例必须在 20%–300% 之间。")
        return (
            PROJECTION_MODES[projection_label],
            sample_rotation_to_camera(rotation_x, rotation_y, rotation_z),
            zoom_percent / 100.0,
        )

    def _set_view_controls_from_preset(self, name: str) -> None:
        preset = VIEW_PRESETS.get(name)
        if preset is None:
            return
        if self._view_update_job is not None:
            self.after_cancel(self._view_update_job)
            self._view_update_job = None
        self._syncing_view_controls = True
        try:
            self.view_preset_var.set(name)
            self.projection_var.set(preset.projection)
            self.rotation_x_var.set(preset.rotation_x)
            self.rotation_y_var.set(preset.rotation_y)
            self.rotation_z_var.set(preset.rotation_z)
            self.view_zoom_var.set(preset.zoom_percent)
        finally:
            self._syncing_view_controls = False

    def apply_view_preset(self, _event: object | None = None) -> None:
        name = self.view_preset_var.get()
        if name not in VIEW_PRESETS:
            return
        self._set_view_controls_from_preset(name)
        if self.ax is not None:
            self._apply_numeric_view(update_status=False)
        self.status_var.set(f"已切换视角：{name}")

    def _run_scheduled_view_update(self) -> None:
        self._view_update_job = None
        self._apply_numeric_view()

    def _apply_numeric_view(self, update_status: bool = True) -> None:
        if self.ax is None:
            return
        if self._view_update_job is not None:
            self.after_cancel(self._view_update_job)
            self._view_update_job = None
        try:
            projection, camera, view_zoom = self._read_numeric_view()
        except (ValueError, tk.TclError) as exc:
            self.status_var.set(str(exc))
            return
        self.ax.set_proj_type(projection)
        if self._current_box_aspect is not None:
            self.ax.set_box_aspect(self._current_box_aspect, zoom=view_zoom)
        self.ax.view_init(elev=camera[0], azim=camera[1], roll=camera[2])
        self.canvas.draw_idle()
        if update_status:
            projection_name = self.projection_var.get().replace("投影", "")
            self.status_var.set(
                f"定量视角：X {self.rotation_x_var.get():g}° · "
                f"Y {self.rotation_y_var.get():g}° · Z {self.rotation_z_var.get():g}° · "
                f"{projection_name} · {self.view_zoom_var.get():g}%"
            )

    def _on_view_interaction_end(self, event: object) -> None:
        if self.ax is None or getattr(event, "inaxes", None) is not self.ax:
            return
        button = getattr(event, "button", None)
        if getattr(button, "value", button) != 1:
            return
        rotation = camera_to_sample_rotation(
            float(self.ax.elev), float(self.ax.azim), float(self.ax.roll)
        )
        self._syncing_view_controls = True
        try:
            self.view_preset_var.set(CUSTOM_VIEW_NAME)
            self.rotation_x_var.set(round(rotation[0], 2))
            self.rotation_y_var.set(round(rotation[1], 2))
            self.rotation_z_var.set(round(rotation[2], 2))
        finally:
            self._syncing_view_controls = False
        self.status_var.set("已同步拖动后的样品旋转角度")

    def _on_scroll_zoom(self, event: object) -> None:
        if self.ax is None or getattr(event, "inaxes", None) is not self.ax:
            return
        direction = getattr(event, "button", None)
        step = float(getattr(event, "step", 0.0) or 0.0)
        factor = 1.1 if direction == "up" or step > 0 else 1.0 / 1.1
        try:
            current = float(self.view_zoom_var.get())
        except (ValueError, tk.TclError):
            current = 100.0
        new_zoom = min(300.0, max(20.0, current * factor))
        self.view_zoom_var.set(round(new_zoom, 1))
        self._apply_numeric_view()

    def reset_view(self) -> None:
        self.set_view_preset(DEFAULT_VIEW_NAME)
        self.status_var.set("视角已重置")

    def export_image(self) -> None:
        if self.raw_data is None or self.ax is None:
            messagebox.showinfo("没有图像", "请先读取数据并生成预览。", parent=self)
            return
        selected = filedialog.asksaveasfilename(
            title="导出当前 3D 视角",
            defaultextension=".png",
            initialfile="infrared_3d_illustration.png",
            filetypes=[
                ("PNG 图像", "*.png"),
                ("TIFF 图像", "*.tif *.tiff"),
            ],
        )
        if not selected:
            return

        try:
            dpi = int(self.dpi_var.get())
            if not 72 <= dpi <= 1200:
                raise ValueError("导出分辨率必须在 72–1200 DPI 之间。")
            output = save_figure_image(
                self.figure,
                selected,
                dpi,
                transparent_background=bool(self.export_transparent_var.get()),
            )
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc), parent=self)
            self.status_var.set("导出失败")
            return

        source_copy: Path | None = None
        source_copy_error: Exception | None = None
        if self.export_cropped_source_var.get():
            try:
                cropped_source = self._crop_current_source()
                source_copy = save_cropped_source_copy(
                    cropped_source,
                    self.source_name,
                    output,
                )
            except Exception as exc:
                source_copy_error = exc

        background_note = "透明背景" if self.export_transparent_var.get() else "预览背景"
        if source_copy_error is not None:
            self.status_var.set(f"已导出图像，但源数据副本失败：{output.name}")
            messagebox.showwarning(
                "图像已导出",
                f"图像已保存到：\n{output}\n\n"
                f"裁剪源数据副本保存失败：\n{source_copy_error}",
                parent=self,
            )
            return

        copy_note = f"\n\n裁剪源数据副本：\n{source_copy}" if source_copy else ""
        self.status_var.set(
            f"已导出：{output.name}（{background_note}）"
            + (f"；数据：{source_copy.name}" if source_copy else "")
        )
        messagebox.showinfo(
            "导出完成",
            f"图像已保存到：\n{output}{copy_note}",
            parent=self,
        )


def _enable_high_dpi() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def main() -> None:
    _enable_high_dpi()
    os.environ.setdefault("MPLBACKEND", "TkAgg")
    app = Infrared3DApp()
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        app.after_idle(lambda: app.load_path(sys.argv[1]))
    app.mainloop()


if __name__ == "__main__":
    main()
