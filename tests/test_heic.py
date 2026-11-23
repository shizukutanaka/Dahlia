"""HEIC/HEIF support tests (W87, run with the Dahlia venv python).

  .venv/bin/python -m unittest tests.test_heic

HEIC is the iPhone default format. The workers decode it through
pillow-heif (registered in workers/common.py); the Rust shell cannot
decode it and works from JPEG proxies written by
workers/heif_convert.py. These tests pin: plain decode, both ways a
photo's rotation can be stored (the container's irot property —
what iPhones write — and an EXIF-only tag), the face worker's
detection over a HEIC photo, the error shown when the decoder
component is missing, and the proxy conversion script.

Skipped when pillow-heif is not installed; the face test also needs
the detection model, like FaceWorkerTest.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "workers"))
sys.path.insert(0, str(ROOT / "workers" / "face"))

import importlib.util  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


import common as workers_common  # noqa: E402

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HAVE_HEIF = True
except Exception:
    HAVE_HEIF = False


def _pattern_image():
    """Stored pixels 4x2: top row red, bottom row black.

    With orientation 6 ("rotate 90 deg CW to display") the upright
    image is 2x4 and the red row becomes its right-hand column.
    """
    from PIL import Image

    img = Image.new("RGB", (4, 2), (0, 0, 0))
    for x in range(4):
        img.putpixel((x, 0), (255, 0, 0))
    return img


def _orientation_exif(value=6):
    from PIL import Image

    exif = Image.Exif()
    exif[274] = value
    return exif


def _assert_upright_pattern(case, img):
    case.assertEqual(img.size, (2, 4))
    r, g, b = img.getpixel((1, 2))
    case.assertTrue(r > 150 and r > g + 80, f"expected red, got {(r, g, b)}")
    r2, g2, b2 = img.getpixel((0, 2))
    case.assertTrue(r2 < 80, f"expected black, got {(r2, g2, b2)}")


@unittest.skipUnless(HAVE_HEIF, "pillow-heif not installed")
class HeicDecodeTest(unittest.TestCase):
    def test_heic_and_heif_extensions_decode(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            src = Image.open(ROOT / "tests" / "samples" / "einstein_a.jpg")
            for ext in ("heic", "heif"):
                path = Path(tmp) / f"photo.{ext}"
                src.save(path, quality=90)
                loaded = workers_common.load_image_rgb(str(path))
                self.assertEqual(loaded.size, src.size)
                self.assertEqual(loaded.mode, "RGB")

    def test_orientation_in_container_property(self):
        """iPhone-style file: rotation as the container's irot
        property (written when the EXIF travels in image info).
        libheif applies it while decoding."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "iphone.heic"
            img = _pattern_image()
            img.info["exif"] = _orientation_exif().tobytes()
            img.save(path, quality=-1, chroma=444)
            self.assertIn(b"irot", path.read_bytes()[: path.stat().st_size])
            loaded = workers_common.load_image_rgb(str(path))
            _assert_upright_pattern(self, loaded)

    def test_orientation_in_exif_only(self):
        """EXIF-only file (no irot): libheif leaves the pixels
        sideways and pillow-heif reports the tag via
        info["original_orientation"]; load_image_rgb must apply it."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exif_only.heic"
            _pattern_image().save(path, exif=_orientation_exif(), quality=-1, chroma=444)
            head = path.read_bytes()
            mdat = head.find(b"mdat")
            self.assertNotIn(b"irot", head if mdat < 0 else head[:mdat])
            loaded = workers_common.load_image_rgb(str(path))
            _assert_upright_pattern(self, loaded)

    def test_no_orientation_loads_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.heic"
            _pattern_image().save(path, quality=-1, chroma=444)
            loaded = workers_common.load_image_rgb(str(path))
            self.assertEqual(loaded.size, (4, 2))
            r, g, b = loaded.getpixel((0, 0))
            self.assertTrue(r > 150, f"expected red, got {(r, g, b)}")


class HeicMissingComponentTest(unittest.TestCase):
    def test_missing_pillow_heif_gives_actionable_error(self):
        with mock.patch.object(workers_common, "_heif_ready", lambda: False):
            with self.assertRaises(RuntimeError) as ctx:
                workers_common.load_image_rgb("photo.heic")
        self.assertIn("pillow-heif", str(ctx.exception))


@unittest.skipUnless(HAVE_HEIF, "pillow-heif not installed")
class HeifConvertScriptTest(unittest.TestCase):
    def test_batch_conversion_and_per_file_errors(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "iphone.heic"
            # A larger pattern than _pattern_image: the proxy is a
            # quality-92 JPEG whose chroma subsampling would bleed
            # across a 2-pixel-wide test image.
            img = Image.new("RGB", (40, 20), (0, 0, 0))
            for y in range(10):
                for x in range(40):
                    img.putpixel((x, y), (255, 0, 0))
            img.info["exif"] = _orientation_exif().tobytes()
            img.save(src, quality=-1, chroma=444)
            proxy = tmp / "proxies" / "1.jpg"
            bogus = tmp / "missing.heic"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "workers" / "heif_convert.py"),
                    str(src),
                    str(proxy),
                    str(bogus),
                    str(tmp / "proxies" / "2.jpg"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            lines = [json.loads(line) for line in proc.stdout.splitlines()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0], {"src": str(src), "ok": True})
            self.assertEqual(lines[1]["src"], str(bogus))
            self.assertFalse(lines[1]["ok"])
            # The proxy is an upright JPEG whose pixels match what
            # the workers see, with no orientation tag left to
            # apply a second time. Orientation 6 turns the stored
            # top half into the right-hand half; sample the middle
            # of each half, away from the JPEG-blurred boundary.
            converted = Image.open(proxy)
            self.assertEqual(converted.format, "JPEG")
            rgb = converted.convert("RGB")
            self.assertEqual(rgb.size, (20, 40))
            r, g, b = rgb.getpixel((15, 20))
            self.assertTrue(r > 150 and r > g + 60, f"expected red, got {(r, g, b)}")
            r2, g2, b2 = rgb.getpixel((5, 20))
            self.assertTrue(r2 < 80, f"expected black, got {(r2, g2, b2)}")
            self.assertIn(converted.getexif().get(274), (None, 1))


@unittest.skipUnless(HAVE_HEIF, "pillow-heif not installed")
@unittest.skipUnless(
    (ROOT / "workers" / "face" / "models" / "face_detection_yunet_2023mar.onnx").exists(),
    "face model not downloaded",
)
class HeicFaceWorkerTest(unittest.TestCase):
    def test_detects_face_in_heic_photo(self):
        from PIL import Image

        face_worker = _load("face_worker", ROOT / "workers" / "face" / "worker.py")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "einstein.heic"
            Image.open(ROOT / "tests" / "samples" / "einstein_a.jpg").save(
                path, quality=90
            )
            faces = face_worker.detect(str(path))
            self.assertEqual(len(faces), 1)
            self.assertGreater(faces[0]["confidence"], 0.5)


if __name__ == "__main__":
    unittest.main()
