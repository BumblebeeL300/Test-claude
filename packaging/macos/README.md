# Building NIM Lab Studio for macOS

This folder turns the `nimlab.studio` package into a double-clickable **NIM Lab Studio.app** (and an
optional `.dmg`) with [PyInstaller](https://pyinstaller.org).

> **Status: not run on macOS.** These files were written and checked on Linux. What was verified
> there: the PyInstaller spec builds and collects all data files, the frozen binary starts, serves
> the UI and the API, and runs a complete simulated Forge build (including the bundled `pytest`
> re-exec described below). What was **not** verified: `build_app.sh` end to end, the `BUNDLE`
> step, `iconutil`, `codesign`, `hdiutil`, the pywebview window on Cocoa, the Keychain backend and
> Gatekeeper behaviour. Expect to fix small things on the first real run and please send them back.

## Requirements

* macOS 12 or newer, Apple silicon or Intel (build on the architecture you want to ship, or see
  *Universal builds*).
* Python 3.10 or newer from python.org or Homebrew (a universal2 python.org build if you want
  `universal2`).
* Xcode command line tools for `codesign` and `iconutil`: `xcode-select --install`.
* Network access for `pip` (the script creates its own virtualenv in `packaging/macos/.build-venv`).

## Build

```bash
git clone https://github.com/BumblebeeL300/Test-claude.git && cd Test-claude
packaging/macos/build_app.sh            # -> packaging/macos/dist/NIM Lab Studio.app
packaging/macos/build_app.sh --dmg      # also writes dist/NIM-Lab-Studio-<version>.dmg
open "packaging/macos/dist/NIM Lab Studio.app"
```

Options: `--no-sign` (skip codesigning), `--clean` (delete `build/`, `dist/` and the build venv),
`--python /path/to/python3`, `--no-venv` (use the current environment), `-h`.
Environment: `NIMLAB_TARGET_ARCH=arm64|x86_64|universal2`, `NIMLAB_CODESIGN_IDENTITY="Developer ID
Application: Your Name (TEAMID)"`.

What the script does:

1. creates `.build-venv` and installs `nimlab[desktop]` (adds `pywebview` and `keyring`),
   `pyinstaller`, `pillow` and `pytest` (Forge runs your project's tests with the bundled pytest);
2. `make_icon.py` draws a 1024 px icon (acid-chartreuse to magenta "N" on deep ink) and builds
   `AppIcon.icns` with `iconutil` (without Pillow it still writes a fallback `AppIcon.png`);
3. `pyinstaller nimlab_studio.spec` builds a windowed one-dir app;
4. ad-hoc `codesign` (or your Developer ID when `NIMLAB_CODESIGN_IDENTITY` is set);
5. optional DMG with `hdiutil` (app plus an `Applications` shortcut).

To run only PyInstaller (for example in CI): `pyinstaller --noconfirm --clean
packaging/macos/nimlab_studio.spec`.

## What is in the bundle

| Piece | How |
| --- | --- |
| Web UI (`nimlab/studio/web`), Atlas UI, Council/Arena report templates, Arena suites and `exemplars.json`, `core/data` | `collect_data_files("nimlab")` in the spec |
| App plugins (`nimlab.council.cli`, ...), uvicorn internals, `_pytest` | `collect_submodules(...)` hidden imports |
| `pywebview` Cocoa backend, `keyring` macOS backend | hidden imports (installed by the `desktop` extra) |
| `Info.plist` | `LSMinimumSystemVersion 12.0`, `NSHighResolutionCapable`, `NSRequiresAquaSystemAppearance = false` (supports dark mode; the app also forces dark appearance at runtime), `NSAppTransportSecurity.NSAllowsLocalNetworking` (the window loads `http://127.0.0.1:<port>`), bundle id `dev.nimlab.studio` |

### Why `entry.py` looks odd

Forge's sandbox runs your project's tests as `[sys.executable, "-m", "pytest", ...]`. Inside a
bundle `sys.executable` is the app itself, so a naive build would open a second Studio window for
every test run. `entry.py` calls `nimlab.studio.frozen.maybe_run_interpreter`, which makes the app act
like `python` for exactly `-m <module>` and `-B script.py`, and otherwise starts Studio. (This path was
exercised on Linux with a frozen binary: a full simulated build ran real pytest through it.)

## Where data lives

The app does not need a working directory. Everything is under `~/NIMLab` (`Projects/`, `atlas.db`,
`arena/runs/`, `studio_history.json`). The API key goes into the **macOS Keychain** (service `nimlab`,
account `nvidia_api_key`) through `keyring`; if the Keychain cannot be used it falls back to
`~/.config/nimlab/credentials.json` (mode 0600). To start from scratch delete `~/NIMLab` and the Keychain
item "nimlab".

## Gatekeeper, signing and sharing

**Your own Mac.** An ad-hoc signature (`codesign --force --deep -s -`, what the script does by default)
is enough for an app you built yourself: it launches normally because nothing downloaded it.

**Another Mac (a DMG you sent or downloaded).** The quarantine flag makes Gatekeeper refuse an ad-hoc
signed app ("cannot be opened because the developer cannot be verified"). Either:

* right-click the app, choose **Open**, then **Open** again (once per app), or
* remove the flag: `xattr -dr com.apple.quarantine "/Applications/NIM Lab Studio.app"`.

On recent macOS versions you may instead need **System Settings > Privacy & Security > Open Anyway**.

**Distributing properly** needs an Apple Developer account:

```bash
export NIMLAB_CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
packaging/macos/build_app.sh --dmg          # signs with hardened runtime + entitlements.plist
xcrun notarytool submit packaging/macos/dist/NIM-Lab-Studio-*.dmg \
      --keychain-profile "notary-profile" --wait
xcrun stapler staple packaging/macos/dist/NIM-Lab-Studio-*.dmg
spctl -a -vv "packaging/macos/dist/NIM Lab Studio.app"
```

`entitlements.plist` grants `disable-library-validation` and `allow-unsigned-executable-memory`, which
a bundled CPython needs under the hardened runtime. The ad-hoc path deliberately does **not** enable the
hardened runtime (library validation would reject the bundled dylibs).

## Universal builds

`NIMLAB_TARGET_ARCH=universal2 packaging/macos/build_app.sh` only works when *every* binary wheel in
the venv is universal2 (use the python.org universal installer; numpy and pydantic-core publish
universal2 wheels). If it fails with "not a universal2 binary", build once per architecture instead.

## Troubleshooting

* **The app bounces and quits.** Start it from Terminal to see the traceback:
  `"packaging/macos/dist/NIM Lab Studio.app/Contents/MacOS/NIM Lab Studio"`. A missing module usually
  needs adding to `hiddenimports` in `nimlab_studio.spec`.
* **Blank window.** The window loads `http://127.0.0.1:<port>/?t=<token>`. Check the ATS key in
  `Info.plist` and that nothing else (a VPN or proxy tool) blocks loopback.
* **Forge says tests cannot run.** pytest must be installed in the build venv (the script installs
  it); confirm with `"<app>/Contents/MacOS/NIM Lab Studio" -m pytest --version`.
* **Icon missing.** Run `python packaging/macos/make_icon.py` (needs Pillow) before PyInstaller, or use
  `build_app.sh`, which does it for you.
