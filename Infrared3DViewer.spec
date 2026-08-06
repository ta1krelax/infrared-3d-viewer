# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        "matplotlib.backends.backend_tkagg",
        "PIL._tkinter_finder",
        "tifffile",
        "imagecodecs",
        "imagecodecs._imcd",
        "imagecodecs._shared_cython",
        "imagecodecs._deflate",
        "imagecodecs._zlib",
        "imagecodecs._lzma",
        "imagecodecs._jpeg8",
        "imagecodecs._jpeg2k",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib.backends.backend_qt",
        "matplotlib.backends.backend_qtagg",
        "matplotlib.backends.backend_webagg",
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Infrared3DViewer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version="version_info.txt",
)
