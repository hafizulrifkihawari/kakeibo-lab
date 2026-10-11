"""A small local web UI: upload a receipt image, see the parsed JSON and an accuracy check.

Run: .venv/bin/python app.py, then open http://localhost:8000
"""

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import ollama
from pydantic import ValidationError

from assess import assess
from ocr_engine import MODEL, check_total, parse_receipt, to_expense

PORT = int(os.environ.get("PORT", "8000"))
INDEX = Path(__file__).with_name("static") / "index.html"
MAX_BYTES = 20 * 1024 * 1024


def _int_header(headers, name):
    value = headers.get(name)
    return int(value) if value and value.lstrip("-").isdigit() else None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
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
