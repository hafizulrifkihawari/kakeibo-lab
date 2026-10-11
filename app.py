"""A small local web UI: upload a receipt image, see the parsed JSON and an accuracy check.
It also reviews the receipts that sync.py downloaded from kakeibo (the inbox).

Run: .venv/bin/python app.py, then open http://localhost:8000
"""

import base64
import hashlib
import json
import re
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import ollama
from pydantic import ValidationError

from assess import assess
from ocr_engine import MODEL, ParsedReceipt, cached_raw, check_total, parse_receipt, postprocess, to_expense
from sync import INBOX, last_sync

PORT = int(os.environ.get("PORT", "8000"))
INDEX = Path(__file__).with_name("static") / "index.html"
MAX_BYTES = 20 * 1024 * 1024
LABELS = Path(__file__).with_name("training-images")
IMAGE_EXTS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/heic": ".heic"}
# The first reading of kakeibo's own scanner, for eval.py --baseline prod.
PROD = LABELS / "prod"
INBOX_PATH = re.compile(r"/inbox/([0-9a-f-]{36})(?:/(parse|accept|discard))?")


def _int_header(headers, name):
    value = headers.get(name)
    return int(value) if value and value.lstrip("-").isdigit() else None


def write_label(stem: str, receipt: ParsedReceipt, image: bytes, ext: str) -> Path:
    """Save a hand-checked receipt for eval.py. An image that is already there stays."""
    LABELS.mkdir(exist_ok=True)
    existing = next((p for p in LABELS.glob(stem + ".*") if p.suffix.lower() != ".json"), None)
    if existing is None:
        (LABELS / (stem + ext)).write_bytes(image)
    label = LABELS / (stem + ".json")
    label.write_text(json.dumps(receipt.model_dump(exclude_none=True), ensure_ascii=False, indent=2))
    return label


def inbox_files(sid: str) -> list:
    return [INBOX / f"{sid}{suffix}" for suffix in (".jpg", ".json", ".prod.json")]


def labeled_count() -> int:
    return len(list(LABELS.glob("*.json")))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/inbox":
            self._inbox_list()
            return
        match = INBOX_PATH.fullmatch(self.path)
        if match and not match.group(2):
            self._inbox_item(match.group(1))
            return
        if self.path != "/":
            self.send_error(404)
            return
        body = INDEX.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/save":
            self._save()
            return
        match = INBOX_PATH.fullmatch(self.path)
        if match and match.group(2):
            {"parse": self._inbox_parse, "accept": self._inbox_accept, "discard": self._inbox_discard}[
                match.group(2)](match.group(1))
            return
        if self.path != "/parse":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        if not 0 < length <= MAX_BYTES:
            self._json(400, {"error": "Send one image of 20 MB or less."})
            return
        image = self.rfile.read(length)
        start = time.monotonic()
        try:
            receipt = parse_receipt(image)
        except ValidationError as e:
            self._json(422, {"error": f"The model output did not match the schema: {e}"})
            return
        except (ollama.ResponseError, ConnectionError) as e:
            self._json(502, {"error": f"Ollama error: {e}. Is 'ollama serve' running?"})
            return
        seconds = round(time.monotonic() - start, 1)
        check_total(receipt, "upload")
        self._json(200, {
            "model": MODEL,
            "seconds": seconds,
            "receipt": receipt.model_dump(exclude_none=True),
            "expense": to_expense(receipt),
            "assessment": assess(
                receipt,
                width=_int_header(self.headers, "X-Image-Width"),
                height=_int_header(self.headers, "X-Image-Height"),
                expected_total=_int_header(self.headers, "X-Expected-Total"),
            ),
        })

    def _save(self):
        """Save the image and its hand-checked receipt as a labeled example for eval.py."""
        length = int(self.headers.get("Content-Length", 0))
        if not 0 < length <= MAX_BYTES * 2:
            self._json(400, {"error": "The request is too large."})
            return
        try:
            data = json.loads(self.rfile.read(length))
            receipt = ParsedReceipt.model_validate(data["receipt"])
            image = base64.b64decode(data["image"])
        except (ValueError, KeyError, ValidationError) as e:
            self._json(422, {"error": f"The receipt is not valid: {e}"})
            return
        ext = IMAGE_EXTS.get(data.get("type", ""), ".jpg")
        stem = re.sub(r"[^\w.-]", "_", Path(data.get("filename") or "").stem)[:80]
        stem = stem or hashlib.sha1(image).hexdigest()[:12]
        label = write_label(stem, receipt, image, ext)
        self._json(200, {"saved": str(label.relative_to(LABELS.parent)), "labeled": labeled_count()})

    # ── Inbox: receipts from kakeibo (sync.py) ──────────────────────────────

    def _inbox_list(self):
        ids = [p.stem for p in sorted(INBOX.glob("*.jpg"), key=lambda p: p.stat().st_mtime)] if INBOX.exists() else []
        last = last_sync()
        self._json(200, {
            "ids": ids,
            "labeled": labeled_count(),
            "lastSync": last and {"time": last["time"].isoformat(), "ok": last["ok"], "line": last["line"]},
        })

    def _inbox_item(self, sid):
        image, label, prod = inbox_files(sid)
        if not image.exists():
            self._json(404, {"error": "This receipt is not in the inbox."})
            return
        qwen = None
        raw = cached_raw(image)
        if raw is not None:
            try:
                qwen = postprocess(raw).model_dump(exclude_none=True)
            except ValidationError:
                pass
        self._json(200, {
            "id": sid,
            "image": base64.b64encode(image.read_bytes()).decode(),
            "label": json.loads(label.read_text()) if label.exists() else None,
            "prod": json.loads(prod.read_text()) if prod.exists() else None,
            "qwen": qwen,
            "model": MODEL,
        })

    def _inbox_parse(self, sid):
        image = inbox_files(sid)[0]
        if not image.exists():
            self._json(404, {"error": "This receipt is not in the inbox."})
            return
        start = time.monotonic()
        try:
            receipt = parse_receipt(image)
        except ValidationError as e:
            self._json(422, {"error": f"The model output did not match the schema: {e}"})
            return
        except (ollama.ResponseError, ConnectionError) as e:
            self._json(502, {"error": f"Ollama error: {e}. Is 'ollama serve' running?"})
            return
        self._json(200, {"model": MODEL, "seconds": round(time.monotonic() - start, 1),
                         "receipt": receipt.model_dump(exclude_none=True)})

    def _inbox_accept(self, sid):
        """The reviewed label becomes a gold label: training-images/<id>.jpg + .json."""
        image, _, prod = inbox_files(sid)
        if not image.exists():
            self._json(404, {"error": "This receipt is not in the inbox."})
            return
        length = int(self.headers.get("Content-Length", 0))
        if not 0 < length <= MAX_BYTES:
            self._json(400, {"error": "The request is too large."})
            return
        try:
            receipt = ParsedReceipt.model_validate(json.loads(self.rfile.read(length))["receipt"])
        except (ValueError, KeyError, ValidationError) as e:
            self._json(422, {"error": f"The receipt is not valid: {e}"})
            return
        label = write_label(sid, receipt, image.read_bytes(), ".jpg")
        if prod.exists():
            PROD.mkdir(exist_ok=True)
            prod.replace(PROD / f"{sid}.json")
        self._remove_from_inbox(sid)
        self._json(200, {"saved": str(label.relative_to(LABELS.parent)), "labeled": labeled_count()})

    def _inbox_discard(self, sid):
        self._remove_from_inbox(sid)
        self._json(200, {"discarded": sid})

    @staticmethod
    def _remove_from_inbox(sid):
        for path in inbox_files(sid):
            path.unlink(missing_ok=True)

    def _json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"Open http://localhost:{PORT}  (model {MODEL})")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
