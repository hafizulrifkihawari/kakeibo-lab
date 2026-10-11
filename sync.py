"""Download receipts scanned in kakeibo into labels/inbox/, for review in the web UI.

kakeibo keeps each scanned photo in its receipt_scans table when scan collection is on for the
user (kakeibo README: "Collect scans for a local model"). This script reads them through wrangler,
writes the files, and then deletes the rows. Review the inbox at http://localhost:8000.

Run:
  .venv/bin/python sync.py               download new scans
  .venv/bin/python sync.py --predict     and let the model read them, for a fast review
  .venv/bin/python sync.py --dry-run     only show what waits
  .venv/bin/python sync.py --install     run every day at 19:00 and at login (launchd)
  .venv/bin/python sync.py --uninstall   stop the daily run

Options: --keep (do not delete the rows), --local (the local kakeibo database, for tests).
Settings (environment or .env): KAKEIBO_DIR (default ../kakeibo), KAKEIBO_D1 (default jp-expense).
"""

import argparse
import base64
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, get_args

ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "labels"
INBOX = LABELS / "inbox"
LOGS = ROOT / "logs"
SYNC_LOG = LOGS / "sync.log"
JOB = "com.kakeibo-lab.sync"
# Names of the daily job in earlier versions. --install and --uninstall remove them.
OLD_JOBS = ["com.receipt-parser.sync"]
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{JOB}.plist"
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
# A scan whose expense never arrived (the save failed) is deleted after this time.
ORPHAN_AFTER = timedelta(days=7)
# The daily job lets the model read only in the evening and at night, so a login run in the
# morning stays light.
EVENING_FROM, MORNING_UNTIL = 18, 6


class SyncError(Exception):
    pass


def load_env() -> None:
    """Read KEY=VALUE lines from .env. A variable that is already set wins."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def settings() -> tuple:
    kakeibo = Path(os.environ.get("KAKEIBO_DIR", "../kakeibo")).expanduser()
    if not kakeibo.is_absolute():
        kakeibo = (ROOT / kakeibo).resolve()
    return kakeibo, os.environ.get("KAKEIBO_D1", "jp-expense")


def log(line: str) -> None:
    LOGS.mkdir(exist_ok=True)
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with SYNC_LOG.open("a") as f:
        f.write(f"{stamp} {line}\n")


def notify(message: str) -> None:
    """A macOS notification. The text goes in as an argument, so quotes in it are safe."""
    script = ['on run argv', 'display notification (item 1 of argv) with title "Receipt sync"', 'end run']
    args = ["osascript"] + [a for line in script for a in ("-e", line)] + [message]
    subprocess.run(args, capture_output=True)


def last_sync() -> Optional[dict]:
    """The last run in logs/sync.log: {"time": datetime, "ok": bool, "line": str}."""
    if not SYNC_LOG.exists():
        return None
    lines = SYNC_LOG.read_text().splitlines()
    if not lines:
        return None
    stamp, _, rest = lines[-1].partition(" ")
    try:
        when = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return {"time": when, "ok": rest.startswith("ok"), "line": rest}


class Kakeibo:
    """The kakeibo D1 database, through wrangler. wrangler has no bind parameters: callers put
    only checked values (UUIDs, numbers) in the SQL text."""

    def __init__(self, folder: Path, database: str, remote: bool):
        self.folder, self.database, self.remote = folder, database, remote
        if not (folder / "wrangler.jsonc").exists() and not (folder / "wrangler.toml").exists():
            raise SyncError(f"No kakeibo project in {folder}. Set KAKEIBO_DIR.")
        self.npx = shutil.which("npx")
        if not self.npx:
            raise SyncError("npx not found. Install Node.js.")

    def _wrangler(self, args: List[str]) -> str:
        try:
            done = subprocess.run([self.npx, "wrangler", *args], cwd=self.folder, capture_output=True,
                                  text=True, timeout=300)
        except subprocess.TimeoutExpired:
            raise SyncError("wrangler did not answer in 5 minutes.")
        if done.returncode != 0:
            detail = (done.stderr.strip() or done.stdout.strip()).splitlines()
            raise SyncError("wrangler failed: " + " | ".join(detail[-3:]))
        return done.stdout

    def check_login(self) -> None:
        if not self.remote:
            return
        out = self._wrangler(["whoami"])
        if "not authenticated" in out.lower():
            raise SyncError("wrangler is not logged in. Run: npx wrangler login")

    def query(self, sql: str) -> List[List[dict]]:
        """The rows of each statement in sql."""
        out = self._wrangler(["d1", "execute", self.database, "--remote" if self.remote else "--local",
                              "--json", "--command", sql])
        try:
            return [r.get("results", []) for r in json.loads(out)]
        except (ValueError, AttributeError):
            raise SyncError("wrangler did not return JSON: " + out[:200])


def to_label(expense: dict, items: List[dict], charges: List[dict]):
    """The kakeibo expense as a ParsedReceipt, the label format of eval.py."""
    from pydantic import ValidationError

    from ocr_engine import CategoryId, Charge, ChargeKind, ParsedReceipt, ReceiptItem, clean_item

    parsed = []
    for i in items:
        fields = {"name": i["name"], "price": i["price"], "qty": i["qty"], "unit": i["unit"],
                  "size": i["size"], "sizeUnit": i["size_unit"], "kind": i["kind"]}
        try:
            item = ReceiptItem(**fields)
        except ValidationError:  # A kind or unit that this tool does not know yet.
            item = ReceiptItem(name=i["name"], price=i["price"])
        parsed.append(clean_item(item))
    category = expense["category_id"] if expense["category_id"] in get_args(CategoryId) else "other"
    return ParsedReceipt(
        store=expense["store"] or None,
        date=expense["date"],
        total=expense["amount"],
        categoryId=category,
        items=parsed,
        charges=[Charge(kind=c["kind"] if c["kind"] in get_args(ChargeKind) else "other",
                        name=c["name"][:100], amount=c["amount"]) for c in charges],
    )


def _write(path: Path, data: bytes) -> None:
    """Write to a temporary file, then rename: a stopped run leaves no half file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def sync(db: Kakeibo, dry_run: bool, keep: bool) -> dict:
    db.check_login()
    [scans] = db.query(
        "SELECT s.expense_id, s.engine, s.created_at, length(s.image) AS size, e.store, e.amount, "
        "e.deleted_at, e.id IS NOT NULL AS found, e.user_id = s.user_id AS same_user "
        "FROM receipt_scans s LEFT JOIN expenses e ON e.id = s.expense_id ORDER BY s.created_at")
    counts = {"synced": 0, "waiting": 0, "dropped": 0}
    if dry_run and not scans:
        print("Nothing waits in kakeibo.")
    now = datetime.now(timezone.utc)
    for scan in scans:
        sid = str(scan["expense_id"])
        if not UUID_RE.match(sid):
            print(f"skip: not a UUID: {sid!r}", file=sys.stderr)
            continue
        created = datetime.strptime(scan["created_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if not scan["found"] and now - created < ORPHAN_AFTER:
            # The expense may still wait in the phone's offline queue.
            counts["waiting"] += 1
            print(f"wait: {sid}: the expense is not saved yet")
            continue
        drop = not scan["found"] or scan["deleted_at"] is not None or not scan["same_user"]
        if dry_run:
            what = "drop (expense deleted or missing)" if drop else f"{scan['store']} ¥{scan['amount']}"
            print(f"{sid}  {scan['created_at']}  {round(scan['size'] * 3 / 4 / 1024)} KB  {what}")
            continue
        if drop:
            if not keep:
                db.query(f"DELETE FROM receipt_scans WHERE expense_id = '{sid}'")
            counts["dropped"] += 1
            print(f"drop: {sid}: the expense was deleted")
            continue
        if not (LABELS / f"{sid}.json").exists() and not download(db, sid, scan["engine"]):  # Not reviewed before.
            continue  # Deleted while this run was busy: the next run drops it.
        if not keep:
            db.query(f"DELETE FROM receipt_scans WHERE expense_id = '{sid}'")
        counts["synced"] += 1
        print(f"synced: {sid}  {scan['store']}  ¥{scan['amount']}")
    return counts


def download(db: Kakeibo, sid: str, engine: str) -> bool:
    """Write the photo, the kakeibo label, and the first reading to the inbox. False when the scan is gone."""
    rows, expenses, items, charges = db.query(
        f"SELECT image, draft, created_at FROM receipt_scans WHERE expense_id = '{sid}';"
        f"SELECT date, amount, store, category_id FROM expenses WHERE id = '{sid}';"
        f"SELECT name, price, qty, unit, size, size_unit, kind FROM expense_items "
        f"WHERE expense_id = '{sid}' ORDER BY sort;"
        f"SELECT kind, name, amount FROM expense_charges WHERE expense_id = '{sid}' ORDER BY sort")
    if not rows or not expenses:
        return False
    row = rows[0]
    label = to_label(expenses[0], items, charges)
    INBOX.mkdir(parents=True, exist_ok=True)
    _write(INBOX / f"{sid}.jpg", base64.b64decode(row["image"]))
    _write(INBOX / f"{sid}.prod.json", json.dumps(
        {"engine": engine, "scanned_at": row["created_at"], "receipt": json.loads(row["draft"])},
        ensure_ascii=False, indent=2).encode())
    _write(INBOX / f"{sid}.json", json.dumps(label.model_dump(exclude_none=True), ensure_ascii=False,
                                             indent=2).encode())
    return True


def on_ac_power() -> bool:
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    return "AC Power" in out or not out  # A Mac with no battery prints nothing useful.


def predict(scheduled: bool) -> tuple:
    """Let the model read each inbox image that it did not read yet. Returns (count, why skipped)."""
    import ollama

    from ocr_engine import cached_raw, raw_parse

    images = [p for p in sorted(INBOX.glob("*.jpg")) if cached_raw(p) is None] if INBOX.exists() else []
    if not images:
        return 0, ""
    if scheduled:
        hour = datetime.now().hour
        if MORNING_UNTIL <= hour < EVENING_FROM:
            return 0, "daytime (the model reads from 18:00)"
        if not on_ac_power():
            return 0, "on battery"
    try:
        ollama.list()
    except Exception:
        return 0, "Ollama is not running"
    done = 0
    for image in images:
        try:
            # keep_alive 0: unload the model after each receipt, so its memory is free again.
            raw_parse(image, keep_alive=0 if scheduled else None)
            done += 1
            print(f"read: {image.stem}")
        except Exception as e:  # One bad image must not stop the others.
            print(f"error: {image.name}: {e}", file=sys.stderr)
    return done, ""


def remove_jobs(labels: List[str]) -> None:
    """Stop each launchd job and delete its plist. A job that is not there is skipped."""
    for label in labels:
        subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{label}"], capture_output=True)
        (PLIST.parent / f"{label}.plist").unlink(missing_ok=True)


def install(local: bool) -> None:
    npx, node = shutil.which("npx"), shutil.which("node")
    if not npx or not node:
        raise SyncError("npx or node not found. Install Node.js, then run --install again.")
    kakeibo, database = settings()
    dirs = []
    for found in (npx, node, shutil.which("ollama")):
        if found and str(Path(found).parent) not in dirs:
            dirs.append(str(Path(found).parent))
    env = {"PATH": ":".join(dirs + ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]),
           "KAKEIBO_DIR": str(kakeibo), "KAKEIBO_D1": database}
    if "OLLAMA_MODEL" in os.environ:
        env["OLLAMA_MODEL"] = os.environ["OLLAMA_MODEL"]
    LOGS.mkdir(exist_ok=True)
    job = {
        "Label": JOB,
        "ProgramArguments": [sys.executable, str(ROOT / "sync.py"), "--scheduled"] + (["--local"] if local else []),
        "WorkingDirectory": str(ROOT),
        "EnvironmentVariables": env,
        "StartCalendarInterval": {"Hour": 19, "Minute": 0},
        "RunAtLoad": True,
        "ProcessType": "Background",
        "StandardOutPath": str(LOGS / "launchd.log"),
        "StandardErrorPath": str(LOGS / "launchd.log"),
    }
    domain = f"gui/{os.getuid()}"
    remove_jobs([JOB] + OLD_JOBS)  # Safe to run again.
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    PLIST.write_bytes(plistlib.dumps(job))
    done = subprocess.run(["launchctl", "bootstrap", domain, str(PLIST)], capture_output=True, text=True)
    if done.returncode != 0:
        raise SyncError(f"launchctl bootstrap failed: {done.stderr.strip()}")
    print(f"Installed {PLIST}. It runs now, every day at 19:00, and at login.")
    print(f"Check it: launchctl print {domain}/{JOB} | grep -E 'state|last exit'")
    protected = [Path.home() / d for d in ("Documents", "Desktop", "Downloads")]
    if any(folder.is_relative_to(p) for folder in (ROOT, kakeibo) for p in protected):
        print(f"""
Warning: macOS does not let a background job read ~/Documents, ~/Desktop, or ~/Downloads.
The job fails before it starts (logs/launchd.log: "Operation not permitted"). Do one of these:
  1. Move this folder and kakeibo to a folder such as ~/Developer, then run --install again.
  2. Give Full Disk Access to {Path(sys.executable).resolve()}
     (System Settings > Privacy & Security > Full Disk Access). Do this again after each Python upgrade.""")


def uninstall() -> None:
    remove_jobs([JOB] + OLD_JOBS)
    print("Removed the daily sync job.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predict", action="store_true", help="let the model read new scans after the download")
    ap.add_argument("--scheduled", action="store_true", help=argparse.SUPPRESS)  # The launchd job.
    ap.add_argument("--dry-run", action="store_true", help="only show what waits in kakeibo")
    ap.add_argument("--keep", action="store_true", help="do not delete the rows in kakeibo")
    ap.add_argument("--local", action="store_true", help="use the local kakeibo database")
    ap.add_argument("--install", action="store_true", help="run every day at 19:00 and at login")
    ap.add_argument("--uninstall", action="store_true", help="stop the daily run")
    args = ap.parse_args()
    load_env()

    try:
        if args.install:
            install(args.local)
            return 0
        if args.uninstall:
            uninstall()
            return 0
        kakeibo, database = settings()
        counts = sync(Kakeibo(kakeibo, database, remote=not args.local), args.dry_run, args.keep)
    except SyncError as e:
        print(f"error: {e}", file=sys.stderr)
        if not args.dry_run:
            log(f"error {e}")
        if args.scheduled:
            notify(str(e))
        return 1
    if args.dry_run:
        return 0

    predicted, skipped = (predict(args.scheduled) if args.predict or args.scheduled else (0, ""))
    waiting = len(list(INBOX.glob("*.jpg"))) if INBOX.exists() else 0
    summary = (f"ok synced={counts['synced']} dropped={counts['dropped']} not_saved_yet={counts['waiting']} "
               f"predicted={predicted} inbox={waiting}" + (f" (model skipped: {skipped})" if skipped else ""))
    log(summary)
    print(summary.removeprefix("ok "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
