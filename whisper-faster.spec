# -*- mode: python ; coding: utf-8 -*-
# PyInstaller: pyinstaller whisper-faster.spec
# One folder (faster to start than one file): whisper-faster[.exe] + _internal

import importlib.util

from PyInstaller.utils.hooks import collect_all, collect_data_files

datas, binaries, hiddenimports = [], [], []
for pkg in ("ctranslate2", "faster_whisper", "tokenizers", "onnxruntime"):
    if importlib.util.find_spec(pkg) is None:
        # onnxruntime (--vad_filter) is not in the macOS 10.11 build
        print("Not installed, left out: %s" % pkg)
        continue
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

# the fixes, the speakers and the OpenVINO backend (imported when used)
hiddenimports += ["transcript_fixes", "diarization", "openvino_backend"]
# what openvino-genai needs of the standard library: it's not in this build
# (fasterwhisper_openvino_<system>.zip, extracted next to the executable),
# but it runs with this Python
hiddenimports += ["hashlib", "uuid", "pkgutil", "tracemalloc", "gettext", "http.client",
                  "multiprocessing", "importlib.metadata"]

a = Analysis(
    ["whisper_faster.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["av", "hf_xet", "tkinter", "matplotlib", "IPython", "pytest", "torch", "tensorflow",
              "openvino", "openvino_genai", "openvino_tokenizers", "openvino_telemetry"],
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
