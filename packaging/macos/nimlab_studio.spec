# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for NIM Lab Studio.app (macOS, windowed, one-dir).

Build with ``build_app.sh`` (it generates the icon first). Run directly with:

    pyinstaller --noconfirm --clean packaging/macos/nimlab_studio.spec

Environment knobs: NIMLAB_TARGET_ARCH (arm64 | x86_64 | universal2, default: current machine),
NIMLAB_CODESIGN_IDENTITY (e.g. "Developer ID Application: ..."; default: no signing here).
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

HERE = Path(SPECPATH).resolve()  # noqa: F821  (SPECPATH is injected by PyInstaller)
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nimlab import __version__  # noqa: E402

APP_NAME = "NIM Lab Studio"
BUNDLE_ID = "dev.nimlab.studio"
ICON = HERE / "AppIcon.icns"

# Everything under src/nimlab that is not Python: studio/web (the UI), atlas/ui, council and arena
# report templates, arena suites + exemplars.json, core/data, py.typed.
datas = collect_data_files("nimlab", include_py_files=False)

hiddenimports = (
    collect_submodules("nimlab")  # app plugins are imported by name (importlib), so list them all
    + collect_submodules("uvicorn")
    + collect_submodules("_pytest")  # Forge runs the user's tests with the bundled pytest
    + [
        "pytest",
        "pytest.__main__",  # `<app> -m pytest` (Forge's sandbox) runs this module
        "webview",
        "webview.platforms.cocoa",
        "keyring.backends.macOS",
        "multipart",
        "h11",
        "anyio._backends._asyncio",
    ]
)

excludes = ["tkinter", "matplotlib", "IPython", "notebook", "PyQt5", "PyQt6", "PySide2", "PySide6"]

a = Analysis(
    [str(HERE / "entry.py")],
    pathex=[str(ROOT / "src")],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    console=False,  # windowed
    argv_emulation=False,
    target_arch=os.environ.get("NIMLAB_TARGET_ARCH") or None,
    codesign_identity=os.environ.get("NIMLAB_CODESIGN_IDENTITY") or None,
    entitlements_file=None,
)
coll = COLLECT(  # noqa: F821
    exe, a.binaries, a.datas, strip=False, upx=False, name=APP_NAME
)

app = BUNDLE(  # noqa: F821  (macOS only; ignored on other platforms)
    coll,
    name=f"{APP_NAME}.app",
    icon=str(ICON) if ICON.exists() else None,
    bundle_identifier=BUNDLE_ID,
    version=__version__,
    info_plist={
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleShortVersionString": __version__,
        "CFBundleVersion": __version__,
        "CFBundleIdentifier": BUNDLE_ID,
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        # Support the system dark appearance (the app itself is dark-only and forces darkAqua).
        "NSRequiresAquaSystemAppearance": False,
        "LSApplicationCategoryType": "public.app-category.developer-tools",
        "NSHumanReadableCopyright": "NIM Lab",
        "NSPrincipalClass": "NSApplication",
        # The window talks to the local server over http://127.0.0.1; allow that, nothing else.
        "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
        # Shown if the app is ever allowed to open folders chosen by the user.
        "NSDocumentsFolderUsageDescription": "NIM Lab Studio stores projects in ~/NIMLab.",
    },
)
