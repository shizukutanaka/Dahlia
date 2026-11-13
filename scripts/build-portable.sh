#!/bin/sh
# Assemble the Windows portable package (W90/W91).
#
# The portable folder is a plain assembly of prepared parts; this
# script is the reproducible recipe. Two parts are produced outside
# this repo and must be supplied as directories:
#
#   STUDIO_EXE   Release build of the app for Windows. On a Windows
#                machine: `cd apps/studio && npm run tauri build`,
#                the exe lands in src-tauri/target/release/. The
#                WebView2Loader.dll from the same build output is
#                picked up automatically when it sits next to it.
#   PYTHON_DIR   A Windows embeddable Python 3.12 with the worker
#                dependencies installed (numpy, pillow, onnxruntime;
#                mediapipe for the facemesh component). This is the
#                `python/` folder of a previous portable package.
#
# Optional:
#   DEPTH_MODEL  depth-anything-v2-small.onnx to bundle AI depth.
#   OUT_DIR      Output directory (default: build/portable).
#
# Result: OUT_DIR/Dahlia/ plus OUT_DIR/Dahlia-V<version>-Windows.zip
set -eu
cd "$(dirname "$0")/.."

: "${STUDIO_EXE:?set STUDIO_EXE to the built studio executable}"
: "${PYTHON_DIR:?set PYTHON_DIR to the prepared Windows Python folder}"
[ -f "$STUDIO_EXE" ] || { echo "STUDIO_EXE not found: $STUDIO_EXE" >&2; exit 1; }
[ -d "$PYTHON_DIR" ] || { echo "PYTHON_DIR not found: $PYTHON_DIR" >&2; exit 1; }

./scripts/check-version.sh
VERSION=$(sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' apps/studio/src-tauri/tauri.conf.json | head -1)
OUT_DIR=${OUT_DIR:-build/portable}
PKG="$OUT_DIR/Dahlia"

rm -rf "$PKG"
mkdir -p "$PKG"

# App executable (+ WebView2 loader when built alongside).
cp "$STUDIO_EXE" "$PKG/studio.exe"
EXE_DIR=$(dirname "$STUDIO_EXE")
if [ -f "$EXE_DIR/WebView2Loader.dll" ]; then
    cp "$EXE_DIR/WebView2Loader.dll" "$PKG/"
fi

# Launcher, readme, self-check (templates live in packaging/).
cp packaging/Dahlia.bat packaging/README.txt packaging/check_env.py "$PKG/"

# Workers: code, docs and the bundled ONNX models. __pycache__ and
# the dev venv never ship.
mkdir -p "$PKG/workers"
for w in face embedding clustering space facemesh; do
    mkdir -p "$PKG/workers/$w"
    cp workers/$w/worker.py "$PKG/workers/$w/"
    # Extra data files a worker ships next to its code (facemesh's
    # canonical tessellation) and its models of any format (.onnx,
    # MediaPipe .task).
    for extra in workers/$w/*.json; do
        [ -f "$extra" ] && cp "$extra" "$PKG/workers/$w/"
    done
    if [ -d "workers/$w/models" ]; then
        mkdir -p "$PKG/workers/$w/models"
        cp workers/$w/models/* "$PKG/workers/$w/models/"
    fi
done
cp workers/common.py workers/README.md "$PKG/workers/"

# Python runtime (as supplied) and the models folder for the
# optional depth model.
cp -R "$PYTHON_DIR" "$PKG/python"
mkdir -p "$PKG/models"
cp packaging/models-README.txt "$PKG/models/README.txt"
if [ -n "${DEPTH_MODEL:-}" ]; then
    cp "$DEPTH_MODEL" "$PKG/models/depth-anything-v2-small.onnx"
fi

# Self-check the assembly before zipping: every part the launcher
# and the app look for must be present.
missing=0
for f in studio.exe Dahlia.bat README.txt check_env.py \
         workers/face/worker.py workers/embedding/worker.py \
         workers/clustering/worker.py workers/space/worker.py \
         workers/facemesh/worker.py workers/common.py \
         workers/face/models/det_500m.onnx \
         workers/embedding/models/w600k_mbf.onnx \
         workers/facemesh/models/face_landmarker.task \
         workers/facemesh/tessellation.json \
         python/python.exe models/README.txt; do
    if [ ! -f "$PKG/$f" ]; then
        echo "missing in package: $f" >&2
        missing=1
    fi
done
[ "$missing" -eq 0 ] || exit 1

ZIP="$OUT_DIR/Dahlia-V$VERSION-Windows.zip"
rm -f "$ZIP"
(cd "$OUT_DIR" && zip -qr "Dahlia-V$VERSION-Windows.zip" Dahlia)
echo "built: $PKG"
echo "built: $ZIP"
