"""Shared worker plumbing: the JSON-Lines loop over stdin/stdout.

Each worker module defines handle(request: dict) -> dict and calls
run_loop(handle). Responses always carry the request id and ok flag.

While a request is being handled, the handler may call
report_progress(done, total) to emit an interim progress line
({"id": rid, "progress": {"done": d, "total": t}}) on the same
channel; the final response still ends the request. The Rust side
forwards these to the UI and treats them as a sign of life, so a
long computation should report regularly instead of going silent.
"""
import json
import sys
import traceback
from pathlib import Path

_progress_reporter = None

HEIF_EXTS = {".heic", ".heif"}

_heif_opener_registered = False


def _heif_ready() -> bool:
    """Register pillow-heif's Pillow opener on first use.

    pillow-heif is the one worker dependency that wraps a native
    library (libheif); it is imported lazily so a runtime without
    it still opens every other format, and only HEIC/HEIF photos
    fail — with the explicit message in load_image_rgb.
    """
    global _heif_opener_registered
    if _heif_opener_registered:
        return True
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        _heif_opener_registered = True
    except Exception:
        pass
    return _heif_opener_registered


# EXIF orientation value -> Pillow transpose method that undoes it.
# Same mapping ImageOps.exif_transpose uses for JPEG; the values are
# PIL.Image.Transpose members, resolved where the table is used.


def _heif_upright(img):
    """Return a decoded HEIF image in display orientation.

    libheif applies the container's rotation property (irot) while
    decoding — that is how iPhone photos arrive upright, and
    pillow-heif resets the EXIF orientation tag to 1 accordingly.
    Some files (certain Android cameras, converters) carry the
    rotation only in the EXIF tag; libheif then leaves the pixels
    sideways while pillow-heif reports the tag through
    info["original_orientation"], so apply that rotation here.
    The two cases are told apart by the container itself: the
    irot property lives in the metadata box, which precedes the
    image data (mdat) in the file.
    """
    orientation = img.info.get("original_orientation")
    if orientation in (None, 1):
        return img
    filename = getattr(img, "filename", None)
    if filename:
        try:
            with open(filename, "rb") as fh:
                head = fh.read(1 << 20)
            mdat = head.find(b"mdat")
            meta_region = head if mdat < 0 else head[:mdat]
            if b"irot" in meta_region:
                return img  # rotation already applied while decoding
        except OSError:
            return img
    else:
        return img
    from PIL import Image

    method = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,  # stored 90 deg CCW -> 90 deg CW
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }.get(orientation)
    if method is None:
        return img
    return img.transpose(method)


def report_progress(done: int, total: int) -> None:
    """Report interim progress for the request being handled.

    A no-op outside run_loop (e.g. unit tests calling handle()
    directly), so handlers can call it unconditionally.
    """
    if _progress_reporter is not None:
        _progress_reporter(int(done), int(total))


def load_image_rgb(path):
    """Open an image as RGB with its EXIF orientation applied.

    Phone photos are usually stored rotated with the intended viewing
    orientation kept only in the EXIF tag; without applying it, faces
    lie sideways and detection misses them (W86). Every worker must
    load source photos through this helper so that pixel coordinates
    (bounding boxes, landmarks) agree across detection, embedding and
    preview generation.

    HEIC/HEIF photos (the iPhone default, W87) decode through
    pillow-heif; their orientation handling differs from JPEG and is
    normalized in _heif_upright.
    """
    from PIL import Image, ImageOps

    if Path(path).suffix.lower() in HEIF_EXTS:
        if not _heif_ready():
            raise RuntimeError(
                "HEIC/HEIF を開くための部品（pillow-heif）が入っていないため、"
                "この写真を読み込めません"
            )
        img = Image.open(path)
        if img.format == "HEIF":
            return _heif_upright(img).convert("RGB")
        # A HEIC-named file holding another format: treat as usual.
        return ImageOps.exif_transpose(img).convert("RGB")
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def run_loop(handle):
    global _progress_reporter
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = req.get("id")

        def _report(done, total, _rid=rid):
            out.write(json.dumps({"id": _rid, "progress": {"done": done, "total": total}}) + "\n")
            out.flush()

        _progress_reporter = _report
        try:
            resp = handle(req)
            resp = {"id": rid, "ok": True, **resp}
        except Exception as e:  # keep the session alive on bad input
            traceback.print_exc(file=sys.stderr)
            resp = {"id": rid, "ok": False, "error": str(e)}
        finally:
            _progress_reporter = None
        out.write(json.dumps(resp) + "\n")
        out.flush()
