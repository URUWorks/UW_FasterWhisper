# -*- mode: python ; coding: utf-8 -*-
# PyInstaller: pyinstaller whisper-faster.spec
# One folder (faster to start than one file): whisper-faster[.exe] + _internal

from PyInstaller.utils.hooks import collect_all, collect_data_files

datas, binaries, hiddenimports = [], [], []
for pkg in ("ctranslate2", "faster_whisper", "tokenizers", "onnxruntime"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

a = Analysis(
    ["whisper_faster.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["av", "hf_xet", "tkinter", "matplotlib", "IPython", "pytest", "torch", "tensorflow"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="whisper-faster",
    console=True,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="whisper-faster", upx=False)
