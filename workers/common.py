"""Shared worker plumbing: the JSON-Lines loop over stdin/stdout.

Each worker module defines handle(request: dict) -> dict and calls
run_loop(handle). Responses always carry the request id and ok flag.
"""
import json
import sys
import traceback


def load_image_rgb(path):
    """Open an image as RGB with its EXIF orientation applied.

    Phone photos are usually stored rotated with the intended viewing
    orientation kept only in the EXIF tag; without applying it, faces
    lie sideways and detection misses them (W86). Every worker must
    load source photos through this helper so that pixel coordinates
    (bounding boxes, landmarks) agree across detection, embedding and
    preview generation.
    """
    from PIL import Image, ImageOps

    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def run_loop(handle):
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
        try:
            resp = handle(req)
            resp = {"id": rid, "ok": True, **resp}
        except Exception as e:  # keep the session alive on bad input
            traceback.print_exc(file=sys.stderr)
            resp = {"id": rid, "ok": False, "error": str(e)}
        out.write(json.dumps(resp) + "\n")
        out.flush()
