from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

import numpy as np

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LightSource, to_rgb
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from core import (
    apply_immersion_absorption,
    crop_temperature,
    downsample_grid,
    load_temperature_file,
    make_demo_data,
    refine_temperature_grid,
)


APP_TITLE = "红外温度 · 3D 浸泡示意图"
CROP_LABELS = {
    "中央 1/3（X、Y）": "xy",
    "仅 X 方向中央 1/3": "x",
    "仅 Y 方向中央 1/3": "y",
    "不截取": "none",
}
MAX_REFINEMENT_LEVEL = 6
MAX_REFINED_POINTS = 90_000

DEFAULT_VIEW_NAME = "常规透视"
VIEW_PRESETS: dict[str, tuple[float, float, float]] = {
    DEFAULT_VIEW_NAME: (28.0, -55.0, 0.0),
    "立方体 · 面（正视）": (0.0, -90.0, 0.0),
    "立方体 · 棱（双面）": (0.0, -45.0, 0.0),
    "立方体 · 角（等轴测）": (35.264, -45.0, 0.0),
    "反向等轴测": (35.264, 135.0, 0.0),
    "俯视（XY）": (90.0, -90.0, 0.0),
    "仰视（XY）": (-90.0, -90.0, 0.0),
    "后视": (0.0, 90.0, 0.0),
    "左视": (0.0, 0.0, 0.0),
    "右视": (0.0, 180.0, 0.0),
    "高角度透视": (55.0, -45.0, 0.0),
    "低角度透视": (15.0, -55.0, 0.0),
}

# Matplotlib's automatic 3-D painter sorting works at collection level.  A
# sample collection containing tall peaks can therefore be drawn after the
# (geometrically higher) water surface and hide/tarnish the translucent water
# in only part of the image.  The renderer uses a fixed opaque-to-transparent
# pass order instead: solid first, then every face of the water volume.
SAMPLE_SURFACE_ZORDER = 10
SAMPLE_WALL_ZORDER = 11
WATER_BOTTOM_ZORDER = 20
WATER_SIDE_ZORDER = 21
WATER_TOP_ZORDER = 22
WATER_EDGE_ZORDER = 23


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
        self.ax = None
        self._update_job: str | None = None
        self._color_buttons: dict[str, tk.Button] = {}
        self.refinement_level = 0

        self.crop_var = tk.StringVar(value="中央 1/3（X、Y）")
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
        self.dpi_var = tk.IntVar(value=300)
        self.export_transparent_var = tk.BooleanVar(value=True)
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
        ttk.Combobox(
            controls,
            textvariable=self.crop_var,
            values=list(CROP_LABELS),
            state="readonly",
        ).grid(row=row, column=0, sticky="ew", pady=(4, 10))
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

        row = self._section_title(controls, row, "显示与导出")
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

        view_wrapper = ttk.Frame(controls, style="Panel.TFrame")
        view_wrapper.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        view_wrapper.columnconfigure(1, weight=1)
        ttk.Label(view_wrapper, text="视角预设", style="Panel.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 12)
        )
        view_box = ttk.Combobox(
            view_wrapper,
            textvariable=self.view_preset_var,
            values=tuple(VIEW_PRESETS),
            state="readonly",
            width=22,
        )
        view_box.grid(row=0, column=1, sticky="ew")
        view_box.bind("<<ComboboxSelected>>", self.apply_view_preset)
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
            text="按住鼠标左键拖动旋转 · 滚轮缩放 · 导出保留当前视角",
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
            self.crop_var,
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

    def _schedule_update(self, *_args: object) -> None:
        if self.raw_data is None:
            return
        if self._update_job is not None:
            self.after_cancel(self._update_job)
        self._update_job = self.after(350, self.update_plot)

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
            cropped = crop_temperature(
                self.raw_data, CROP_LABELS[self.crop_var.get()]
            )
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
        self.refinement_level = 0
        self.refinement_info_var.set("细化次数：0")
        self.source_name = str(Path(selected))
        self.source_format = loaded.source_format
        self.title(f"{APP_TITLE} · {Path(selected).name}")
        self.source_var.set(
            f"{Path(selected).name}\n{loaded.values.shape[1]} × {loaded.values.shape[0]} px · {loaded.source_format}"
        )
        self.status_var.set("已读取数据")
        self.update_plot(reset_camera=True)
        return True

    def load_demo(self) -> None:
        self.raw_data = make_demo_data()
        self.refinement_level = 0
        self.refinement_info_var.set("细化次数：0")
        self.source_name = "内置示例"
        self.source_format = "二维温度矩阵"
        self.title(APP_TITLE)
        self.source_var.set(
            f"内置示例\n{self.raw_data.shape[1]} × {self.raw_data.shape[0]} px · 二维温度矩阵"
        )
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

        try:
            settings = self._read_settings()
            cropped = crop_temperature(self.raw_data, settings.crop_mode)
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

        camera = VIEW_PRESETS[DEFAULT_VIEW_NAME]
        if reset_camera:
            self.view_preset_var.set(DEFAULT_VIEW_NAME)
        elif self.ax is not None:
            camera = (
                float(getattr(self.ax, "elev", camera[0])),
                float(getattr(self.ax, "azim", camera[1])),
                float(getattr(self.ax, "roll", camera[2])),
            )

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
        # Transparent 3-D collections must be composited after the opaque
        # sample.  Automatic collection-level sorting is unstable for a water
        # plane intersected by many peaks, so use the explicit material order
        # defined above.  It remains stable while the user rotates the view.
        self.ax = self.figure.add_subplot(111, projection="3d", computed_zorder=False)
        self.ax.set_facecolor("#f4f6f8")
        sample_base = self._draw_sample(
            xx,
            yy,
            z_small,
            raw_small,
            result.waterline_temperature,
            settings,
        )
        water_bottom, water_top = self._draw_glass_water(
            xx,
            yy,
            result.waterline_temperature,
            plot_data,
            raw_small,
            sample_base,
            settings,
        )

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
        self.ax.set_zlim(z_min, z_max)
        self.ax.set_box_aspect((display_x, display_y, z_box))
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

        self.ax.plot_surface(
            xx,
            yy,
            z,
            facecolors=rgba,
            linewidth=0,
            antialiased=False,
            shade=False,
            rstride=1,
            cstride=1,
            zorder=SAMPLE_SURFACE_ZORDER,
        )

        if settings.show_grid:
            row_stride = max(1, int(np.ceil(z.shape[0] / settings.grid_count)))
            column_stride = max(1, int(np.ceil(z.shape[1] / settings.grid_count)))
            self.ax.plot_wireframe(
                xx,
                yy,
                z + max(float(np.ptp(z)), 0.01) * 0.001,
                rstride=row_stride,
                cstride=column_stride,
                color=(0.08, 0.10, 0.12, 0.38),
                linewidth=0.42,
                zorder=SAMPLE_WALL_ZORDER + 1,
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

        wall_collection = Poly3DCollection(
            walls,
            facecolors=wall_colors,
            edgecolors=(0.15, 0.17, 0.19, 0.12),
            linewidths=0.15,
            zorder=SAMPLE_WALL_ZORDER,
        )
        self.ax.add_collection3d(wall_collection)
        bottom_rgb = np.clip(below_base * 0.62, 0.0, 1.0)
        bottom = Poly3DCollection(
            [[
                (xx[0, 0], yy[0, 0], base_z),
                (xx[0, -1], yy[0, -1], base_z),
                (xx[-1, -1], yy[-1, -1], base_z),
                (xx[-1, 0], yy[-1, 0], base_z),
            ]],
            facecolors=[(*bottom_rgb, settings.below_alpha)],
            edgecolors="none",
            zorder=SAMPLE_WALL_ZORDER,
        )
        self.ax.add_collection3d(bottom)
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
            self.ax.plot_surface(
                water_xx,
                water_yy,
                water_z_visible,
                facecolors=water_rgba,
                linewidth=0,
                antialiased=True,
                shade=False,
                rstride=1,
                cstride=1,
                zorder=WATER_TOP_ZORDER,
            )

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
        side_collection = Poly3DCollection(
            side_faces,
            facecolors=side_colors,
            edgecolors="none",
            linewidths=0,
            zorder=WATER_SIDE_ZORDER,
        )
        self.ax.add_collection3d(side_collection)

        bottom_face = Poly3DCollection(
            [[
                (water_x[0], water_y[0], water_bottom),
                (water_x[-1], water_y[0], water_bottom),
                (water_x[-1], water_y[-1], water_bottom),
                (water_x[0], water_y[-1], water_bottom),
            ]],
            facecolors=[(*base_rgb, settings.water_alpha)],
            edgecolors="none",
            zorder=WATER_BOTTOM_ZORDER,
        )
        self.ax.add_collection3d(bottom_face)

        if settings.show_water_edges:
            edge_rgb = np.clip(base_rgb * 0.85 + 0.15, 0.0, 1.0)
            edge_color = (*edge_rgb, settings.water_alpha)
            for edge_x, edge_y, edge_z in perimeter:
                self.ax.plot(
                    edge_x,
                    edge_y,
                    edge_z,
                    color=edge_color,
                    linewidth=0.85,
                    zorder=WATER_EDGE_ZORDER,
                )
            corners = [
                (water_x[0], water_y[0], water_z[0, 0]),
                (water_x[-1], water_y[0], water_z[0, -1]),
                (water_x[-1], water_y[-1], water_z[-1, -1]),
                (water_x[0], water_y[-1], water_z[-1, 0]),
            ]
            for corner_x, corner_y, corner_z in corners:
                self.ax.plot(
                    [corner_x, corner_x],
                    [corner_y, corner_y],
                    [water_bottom, corner_z],
                    color=edge_color,
                    linewidth=0.75,
                    zorder=WATER_EDGE_ZORDER,
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

    def apply_view_preset(self, _event: object | None = None) -> None:
        if self.ax is None:
            return
        name = self.view_preset_var.get()
        camera = VIEW_PRESETS.get(name)
        if camera is None:
            return
        self.ax.view_init(elev=camera[0], azim=camera[1], roll=camera[2])
        self.canvas.draw_idle()
        self.status_var.set(f"已切换视角：{name}")

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

        background_note = "透明背景" if self.export_transparent_var.get() else "预览背景"
        self.status_var.set(f"已导出：{output.name}（{background_note}）")
        messagebox.showinfo("导出完成", f"图像已保存到：\n{output}", parent=self)


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
