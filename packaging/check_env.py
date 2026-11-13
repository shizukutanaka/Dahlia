"""Dahlia portable self-check: verifies the bundled Python runtime can
import everything the face pipeline needs. Writes selfcheck.txt."""
import importlib
import sys

lines = [f"python: {sys.version.split()[0]} ({sys.executable})"]
for mod in ("numpy", "PIL", "onnxruntime"):
    try:
        m = importlib.import_module(mod)
        lines.append(f"{mod}: OK {getattr(m, '__version__', '')}".rstrip())
    except Exception as e:
        lines.append(f"{mod}: FAIL {e}")
try:
    import mediapipe  # noqa: F401

    lines.append("facemesh component (mediapipe): OK")
except Exception:
    lines.append("facemesh component (mediapipe): not installed (optional)")
try:
    import pillow_heif

    lines.append(f"heic support (pillow-heif): OK {pillow_heif.__version__}")
except Exception:
    lines.append("heic support (pillow-heif): not installed (HEIC photos cannot be imported)")

text = "\n".join(lines) + "\n"
try:
    with open("selfcheck.txt", "w", encoding="utf-8") as fh:
        fh.write(text)
except OSError:
    pass
print(text)
