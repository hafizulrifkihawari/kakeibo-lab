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
from typing import get_args

import ollama
from pydantic import ValidationError

from assess import assess
from ocr_engine import (MODEL, CategoryId, ChargeKind, ItemKind, ParsedReceipt, cached_raw, check_total,
                        parse_receipt, postprocess, to_expense)
from eval import labeled, split_of
from sync import INBOX, last_sync

PORT = int(os.environ.get("PORT", "8000"))
INDEX = Path(__file__).with_name("static") / "index.html"
MAX_BYTES = 20 * 1024 * 1024
LABELS = Path(__file__).with_name("labels")
IMAGE_EXTS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/heic": ".heic"}
IMAGE_TYPES = {ext: mime for mime, ext in IMAGE_EXTS.items()} | {".jpeg": "image/jpeg"}
# The first reading of kakeibo's own scanner, for eval.py --baseline prod.
PROD = LABELS / "prod"
INBOX_PATH = re.compile(r"/inbox/([0-9a-f-]{36})(?:/(parse|accept|discard))?")
# A label's file name without the extension, as _save() makes it. No "/" and no leading ".".
LABEL_PATH = re.compile(r"/labels/([A-Za-z0-9_][\w.-]{0,79})(?:/(parse|save|delete))?")


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


def label_files(stem: str) -> tuple:
    """(image or None, label, kakeibo's first reading) of a saved label."""
    image = next((p for p in LABELS.glob(stem + ".*") if p.suffix.lower() in IMAGE_TYPES), None)
    return image, LABELS / f"{stem}.json", PROD / f"{stem}.json"


def labeled_count() -> int:
    return len(list(LABELS.glob("*.json")))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/inbox":
            self._inbox_list()
            return
        if self.path == "/options":  # The choices of the label form.
            self._json(200, {"categories": get_args(CategoryId), "kinds": get_args(ItemKind),
                             "chargeKinds": get_args(ChargeKind)})
            return
        if self.path == "/labels":
            self._labels_list()
            return
        match = INBOX_PATH.fullmatch(self.path)
        if match and not match.group(2):
            self._inbox_item(match.group(1))
            return
        match = LABEL_PATH.fullmatch(self.path)
        if match and not match.group(2) and ".." not in match.group(1):
            self._label_item(match.group(1))
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
        match = LABEL_PATH.fullmatch(self.path)
        if match and match.group(2) and ".." not in match.group(1):
            {"parse": self._label_parse, "save": self._label_save, "delete": self._label_delete}[
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

    # ── Shared by the inbox and the saved labels ────────────────────────────

    def _review(self, sid, image, label, prod):
        """The photo, the label, kakeibo's first reading, and the model's cached reading, for the review form."""
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
            "type": IMAGE_TYPES.get(image.suffix.lower(), "image/jpeg"),
            "label": json.loads(label.read_text()) if label.exists() else None,
            "prod": json.loads(prod.read_text()) if prod.exists() else None,
            "qwen": qwen,
            "model": MODEL,
        })

    def _parse_image(self, image):
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

    def _read_receipt(self):
        """The receipt in the request body, or None after an error reply."""
        length = int(self.headers.get("Content-Length", 0))
        if not 0 < length <= MAX_BYTES:
            self._json(400, {"error": "The request is too large."})
            return None
        try:
            return ParsedReceipt.model_validate(json.loads(self.rfile.read(length))["receipt"])
        except (ValueError, KeyError, ValidationError) as e:
            self._json(422, {"error": f"The receipt is not valid: {e}"})
            return None

    # ── Inbox: receipts from kakeibo (sync.py) ──────────────────────────────

    def _inbox_item(self, sid):
        image, label, prod = inbox_files(sid)
        if not image.exists():
            self._json(404, {"error": "This receipt is not in the inbox."})
            return
        self._review(sid, image, label, prod)

    def _inbox_parse(self, sid):
        image = inbox_files(sid)[0]
        if not image.exists():
            self._json(404, {"error": "This receipt is not in the inbox."})
            return
        self._parse_image(image)

    def _inbox_accept(self, sid):
        """The reviewed label becomes a gold label: labels/<id>.jpg + .json."""
        image, _, prod = inbox_files(sid)
        if not image.exists():
            self._json(404, {"error": "This receipt is not in the inbox."})
            return
        receipt = self._read_receipt()
        if receipt is None:
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

    # ── Saved labels (labels/) ─────────────────────────────────────

    def _labels_list(self):
        rows = []
        for image, label in labeled(LABELS):
            try:
                data = json.loads(label.read_text())
            except ValueError:
                data = {}
            rows.append({"id": label.stem, "store": data.get("store"), "date": data.get("date"),
                         "total": data.get("total"), "items": len(data.get("items") or []),
                         "split": split_of(label.stem), "kakeibo": (PROD / label.name).exists(),
                         "modified": label.stat().st_mtime})
        rows.sort(key=lambda r: r["modified"], reverse=True)
        self._json(200, {"labels": rows})

    def _label_item(self, stem):
        image, label, prod = label_files(stem)
        if image is None or not label.exists():
            self._json(404, {"error": "No saved label with this name."})
            return
        self._review(stem, image, label, prod)

    def _label_parse(self, stem):
        image = label_files(stem)[0]
        if image is None:
            self._json(404, {"error": "No saved label with this name."})
            return
        self._parse_image(image)

    def _label_save(self, stem):
        image, label, _ = label_files(stem)
        if image is None or not label.exists():
            self._json(404, {"error": "No saved label with this name."})
            return
        receipt = self._read_receipt()
        if receipt is None:
            return
        write_label(stem, receipt, b"", image.suffix)  # The image is already there and stays.
        self._json(200, {"saved": str(label.relative_to(LABELS.parent)), "labeled": labeled_count()})

    def _label_delete(self, stem):
        """Delete a saved label, its photo, and kakeibo's first reading."""
        image, label, prod = label_files(stem)
        for path in (image, label, prod):
            if path is not None:
                path.unlink(missing_ok=True)
        self._json(200, {"deleted": stem, "labeled": labeled_count()})

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
