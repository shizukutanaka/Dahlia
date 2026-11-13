#!/usr/bin/env python3
"""One-shot HEIC/HEIF -> JPEG conversion for the Rust shell.

The Rust side reads pixels with the `image` crate (thumbnails and
face crops), which has no HEIC decoder, while the workers decode
HEIC natively through Pillow. Import therefore asks this script to
write a full-resolution JPEG working copy ("proxy") of each HEIC
photo into the app data directory; thumbnails and crops are then
cut from the proxy, whose pixels are exactly what the workers see
(same loader, orientation already applied, no EXIF tag re-written,
so nothing can be rotated twice).

Usage: heif_convert.py SRC DST [SRC DST ...]

Prints one JSON line per pair:
  {"src": SRC, "ok": true}
  {"src": SRC, "ok": false, "error": "..."}
A failing pair never stops the rest of the batch. Exit status is 0
whenever the script itself ran; it is 2 only for a malformed call.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_image_rgb  # noqa: E402


def convert(src: str, dst: str) -> None:
    img = load_image_rgb(src)
    # The orientation is already applied to the pixels; the stored
    # EXIF must not travel with the proxy or a later reader would
    # apply it a second time.
    img.info.pop("exif", None)
    img.info.pop("xmp", None)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "JPEG", quality=92)


def main(argv: list[str]) -> int:
    args = argv[1:]
    if not args or len(args) % 2 != 0:
        print(
            json.dumps(
                {"src": None, "ok": False, "error": "引数が SRC/DST の組になっていません"}
            ),
            flush=True,
        )
        return 2
    for i in range(0, len(args), 2):
        src, dst = args[i], args[i + 1]
        try:
            convert(src, dst)
            print(json.dumps({"src": src, "ok": True}), flush=True)
        except Exception as e:  # report and continue with the batch
            print(json.dumps({"src": src, "ok": False, "error": str(e)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
