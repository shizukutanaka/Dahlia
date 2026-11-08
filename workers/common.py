"""Shared worker plumbing: the JSON-Lines loop over stdin/stdout.

Each worker module defines handle(request: dict) -> dict and calls
run_loop(handle). Responses always carry the request id and ok flag.
"""
import json
import sys
import traceback


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
