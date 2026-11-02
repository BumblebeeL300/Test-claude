#!/usr/bin/env bash
# Build "NIM Lab Studio.app" (and optionally a .dmg) on macOS.
#
#   packaging/macos/build_app.sh [--dmg] [--no-sign] [--clean] [--python PYTHON] [--no-venv]
#
# Steps: create/refresh a build virtualenv -> install nimlab[desktop] + PyInstaller + Pillow ->
# draw the icon -> PyInstaller (nimlab_studio.spec) -> ad-hoc codesign -> optional DMG.
# Output: packaging/macos/dist/NIM Lab Studio.app  (and NIM-Lab-Studio-<version>.dmg with --dmg)
#
# Environment: NIMLAB_TARGET_ARCH=arm64|x86_64|universal2   NIMLAB_CODESIGN_IDENTITY="Developer ID ..."
set -euo pipefail

APP_NAME="NIM Lab Studio"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DIST="$HERE/dist"
WORK="$HERE/build"
VENV="$HERE/.build-venv"

MAKE_DMG=0
SIGN=1
CLEAN=0
USE_VENV=1
PYTHON_BIN="${PYTHON:-python3}"

die() { echo "error: $*" >&2; exit 1; }
say() { printf '\n==> %s\n' "$*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dmg) MAKE_DMG=1 ;;
    --no-sign) SIGN=0 ;;
    --clean) CLEAN=1 ;;
    --no-venv) USE_VENV=0 ;;
    --python) shift; PYTHON_BIN="${1:?--python needs a path}" ;;
    -h|--help) sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

# --- preflight ---------------------------------------------------------------------------------
if [[ "$(uname -s)" != "Darwin" ]]; then
  die "this script builds a macOS .app and must run on macOS (found $(uname -s)).
       PyInstaller cannot cross-compile. Run it on a Mac, or in a macOS CI runner (macos-14)."
fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python not found ('$PYTHON_BIN'). Install Python 3.10+ (python.org or Homebrew)."
"$PYTHON_BIN" - <<'PY' || die "Python 3.10 or newer is required."
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY
if [[ "$MAKE_DMG" == 1 ]]; then command -v hdiutil >/dev/null 2>&1 || die "hdiutil not found (it ships with macOS)."; fi
if [[ "$SIGN" == 1 ]]; then command -v codesign >/dev/null 2>&1 || die "codesign not found: install the Xcode command line tools (xcode-select --install) or pass --no-sign."; fi

# --- environment -------------------------------------------------------------------------------
[[ "$CLEAN" == 1 ]] && { say "Cleaning previous build output"; rm -rf "$DIST" "$WORK" "$VENV"; }

if [[ "$USE_VENV" == 1 ]]; then
  say "Preparing build environment ($VENV)"
  [[ -d "$VENV" ]] || "$PYTHON_BIN" -m venv "$VENV"
  PY="$VENV/bin/python"
else
  PY="$PYTHON_BIN"
fi
"$PY" -m pip install --quiet --upgrade pip
say "Installing nimlab[desktop], PyInstaller and Pillow"
"$PY" -m pip install --quiet "$ROOT[desktop]" "pyinstaller>=6" pillow pytest

# --- icon --------------------------------------------------------------------------------------
say "Drawing the app icon"
"$PY" "$HERE/make_icon.py"

# --- build -------------------------------------------------------------------------------------
say "Running PyInstaller"
"$PY" -m PyInstaller --noconfirm --clean --distpath "$DIST" --workpath "$WORK" "$HERE/nimlab_studio.spec"
APP="$DIST/$APP_NAME.app"
[[ -d "$APP" ]] || die "PyInstaller finished but $APP was not created."

# --- sign --------------------------------------------------------------------------------------
if [[ "$SIGN" == 1 ]]; then
  IDENTITY="${NIMLAB_CODESIGN_IDENTITY:--}"
  if [[ "$IDENTITY" == "-" ]]; then
    say "Ad-hoc signing (local use only; see README for Gatekeeper)"
    # No hardened runtime for ad-hoc: library validation would reject the bundled dylibs.
    codesign --force --deep -s - "$APP"
  else
    say "Signing as: $IDENTITY (hardened runtime, ready for notarization)"
    codesign --force --deep --timestamp --options runtime \
      --entitlements "$HERE/entitlements.plist" -s "$IDENTITY" "$APP"
  fi
  codesign --verify --deep --strict "$APP" && echo "signature OK"
fi

# --- dmg ---------------------------------------------------------------------------------------
if [[ "$MAKE_DMG" == 1 ]]; then
  VERSION="$("$PY" -c 'import nimlab; print(nimlab.__version__)')"
  DMG="$DIST/NIM-Lab-Studio-$VERSION.dmg"
  STAGE="$(mktemp -d)"
  trap 'rm -rf "$STAGE"' EXIT
  say "Creating $DMG"
  cp -R "$APP" "$STAGE/"
  ln -s /Applications "$STAGE/Applications"
  rm -f "$DMG"
  hdiutil create -volname "$APP_NAME" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
  echo "wrote $DMG"
fi

say "Done: $APP"
echo "Open it with:  open \"$APP\""
