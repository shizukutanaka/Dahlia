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

_progress_reporter = None


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
    """
    from PIL import Image, ImageOps

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
