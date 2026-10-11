"""Measure the parser against hand-checked receipts.

A labeled receipt is an image with a .json file of the same name next to it
(training-images/PXL_1.jpg + training-images/PXL_1.json). Save labels from the web UI.

Each receipt is in the dev split or the test split (1 in 5), fixed by its file name.
Tune on dev. Before you keep a change, check that test does not go down.

Run: .venv/bin/python eval.py [DIR] [--split dev|test|all] [--baseline prod] [--out results.json]
  --baseline prod   score kakeibo's own scan (training-images/prod/) instead of the model

Every run is saved in runs/ and compared with the last run of the same kind.
The model's raw answers are cached in .cache/, so a change to clean() or fix() runs fast.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional

import ollama
from pydantic import ValidationError

from ocr_engine import MODEL, ParsedReceipt, cached_raw, parse_receipt

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".heic")
NAME_MATCH = 0.5
RUNS = Path(__file__).with_name("runs")


def norm(s: Optional[str]) -> str:
    return unicodedata.normalize("NFKC", s or "").replace(" ", "").replace("　", "").lower()


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def _match(truth: list, pred: list, key) -> List[tuple]:
    """Pairs of (truth, pred) that match. Each pred is used once."""
    left = list(pred)
    pairs = []
    for t in truth:
        best = max(left, key=lambda p: key(t, p), default=None)
        if best is not None and key(t, best) > 0:
            pairs.append((t, best))
            left.remove(best)
    return pairs


def _item_key(t, p) -> float:
    if t.price != p.price:
        return 0
    s = _similar(t.name, p.name)
    return s if s >= NAME_MATCH else 0


def _f1(n_match: int, n_truth: int, n_pred: int) -> float:
    if n_truth == 0 and n_pred == 0:
        return 1.0
    return 2 * n_match / (n_truth + n_pred)


def compare(truth: ParsedReceipt, pred: ParsedReceipt) -> dict:
    """Field scores (0–1) and a list of error tags for one receipt."""
    errors = []
    fields = {
        "store": _similar(truth.store or "", pred.store or "") >= 0.8,
        "date": truth.date == pred.date,
        "total": truth.total == pred.total,
        "category": truth.categoryId == pred.categoryId,
    }
    for name, ok in fields.items():
        if not ok:
            errors.append(f"{name}: expected {getattr(truth, 'categoryId' if name == 'category' else name)!r}, "
                          f"got {getattr(pred, 'categoryId' if name == 'category' else name)!r}")

    items = _match(truth.items, pred.items, _item_key)
    matched_t = {id(t) for t, _ in items}
    matched_p = {id(p) for _, p in items}
    errors += [f"missing item: {t.name} {t.price}" for t in truth.items if id(t) not in matched_t]
    errors += [f"extra item: {p.name} {p.price}" for p in pred.items if id(p) not in matched_p]

    kinds = [(t, p) for t, p in items if t.kind]
    errors += [f"kind: {t.name} expected {t.kind}, got {p.kind}" for t, p in kinds if t.kind != p.kind]

    charges = _match(truth.charges, pred.charges, lambda t, p: 1 if t.amount == p.amount and t.kind == p.kind else 0)
    matched_t = {id(t) for t, _ in charges}
    matched_p = {id(p) for _, p in charges}
    errors += [f"missing charge: {t.kind} {t.name} {t.amount}" for t in truth.charges if id(t) not in matched_t]
    errors += [f"extra charge: {p.kind} {p.name} {p.amount}" for p in pred.charges if id(p) not in matched_p]

    scores = {k: float(v) for k, v in fields.items()}
    scores["items"] = _f1(len(items), len(truth.items), len(pred.items))
    scores["charges"] = _f1(len(charges), len(truth.charges), len(pred.charges))
    scores["kind"] = sum(t.kind == p.kind for t, p in kinds) / len(kinds) if kinds else 1.0
    return {"scores": scores, "errors": errors}


def labeled(folder: Path) -> List[tuple]:
    pairs = []
    for label in sorted(folder.glob("*.json")):
        image = next((label.with_suffix(e) for e in IMAGE_EXTS + tuple(x.upper() for x in IMAGE_EXTS)
                      if label.with_suffix(e).exists()), None)
        if image:
            pairs.append((image, label))
    return pairs


def _error_type(e: str) -> str:
    return e.split(":")[0]


def split_of(stem: str) -> str:
    """'test' for 1 in 5 receipts, else 'dev'. The file name fixes it, so it never changes."""
    return "test" if int(hashlib.sha1(stem.encode()).hexdigest(), 16) % 5 == 0 else "dev"


def prod_reading(label: Path) -> Optional[ParsedReceipt]:
    """kakeibo's first reading of this receipt (saved by the review page), or None."""
    path = label.parent / "prod" / label.name
    return ParsedReceipt.model_validate(json.loads(path.read_text())["receipt"]) if path.exists() else None


def _version() -> str:
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip()
    sha = git("rev-parse", "--short", "HEAD") or "nogit"
    return sha + ("+changes" if git("status", "--porcelain", "--untracked-files=no") else "")


def _mean(scores: dict) -> float:
    return sum(scores.values()) / len(scores)


def compare_runs(run: dict) -> None:
    """Print the change from the last run of the same kind, on the receipts in both runs."""
    kind = (run["split"], run["baseline"], run["model"])
    old = None
    for path in sorted(RUNS.glob("*.json"), reverse=True):
        prev = json.loads(path.read_text())
        if (prev.get("split"), prev.get("baseline"), prev.get("model")) == kind and prev["time"] != run["time"]:
            old = prev
            break
    if old is None:
        return
    before = {r["image"]: r for r in old["receipts"]}
    common = [r for r in run["receipts"] if r["image"] in before]
    if not common:
        return
    print(f"\nChange from the last run ({old['time']}, {old['version']}) on {len(common)} receipts in both runs:")
    for field in common[0]["scores"]:
        was = sum(before[r["image"]]["scores"].get(field, 0) for r in common) / len(common)
        now = sum(r["scores"][field] for r in common) / len(common)
        mark = "" if abs(now - was) < 0.005 else ("  better" if now > was else "  WORSE")
        print(f"  {field:<9} {was:.0%} → {now:.0%}{mark}")
    worse = [(r["image"], _mean(before[r["image"]]["scores"]), _mean(r["scores"])) for r in common
             if _mean(r["scores"]) < _mean(before[r["image"]]["scores"]) - 0.005]
    for image, was, now in worse:
        print(f"  worse: {image} {was:.0%} → {now:.0%}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", default="training-images")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--baseline", choices=["prod"], help="score kakeibo's own scan instead of the model")
    ap.add_argument("--out", help="Write the full results to this JSON file")
    args = ap.parse_args()

    pairs = [(i, l) for i, l in labeled(Path(args.dir)) if args.split == "all" or split_of(l.stem) == args.split]
    if not pairs:
        print(f"No labeled receipts in the {args.split} split of {args.dir}. "
              "Review the inbox or use 'Save as correct' in the web UI.", file=sys.stderr)
        return 2
    name = "kakeibo scan" if args.baseline else MODEL

    results, totals, error_types, no_prod = [], Counter(), Counter(), 0
    for image, label in pairs:
        truth = ParsedReceipt.model_validate_json(label.read_text())
        start = time.monotonic()
        cached = False
        try:
            if args.baseline:
                pred = prod_reading(label)
                if pred is None:
                    no_prod += 1
                    continue
            else:
                cached = cached_raw(image) is not None
                pred = parse_receipt(image)
        except (ValidationError, ollama.ResponseError, OSError) as e:
            print(f"error: {image.name}: {e}", file=sys.stderr)
            continue
        r = compare(truth, pred)
        if args.baseline:
            del r["scores"]["kind"]  # kakeibo's scan does not read item kinds.
        r.update(image=image.name, split=split_of(label.stem), seconds=round(time.monotonic() - start, 1))
        results.append(r)
        totals.update(r["scores"])
        error_types.update(_error_type(e) for e in r["errors"])
        print(f"{image.name}  {_mean(r['scores']):.0%}  ({'cached' if cached else str(r['seconds']) + ' s'})")
        for e in r["errors"]:
            print(f"    {e}")

    if no_prod:
        print(f"{no_prod} receipts have no kakeibo scan (only receipts reviewed from the inbox have one).")
    if not results:
        return 1
    n = len(results)
    overall = {k: v / n for k, v in totals.items()}
    print(f"\n{name} on {n} receipts ({args.split})")
    for k, v in overall.items():
        print(f"  {k:<9} {v:.0%}")
    print(f"  {'overall':<9} {_mean(overall):.0%}")
    if error_types:
        print("Most common errors: " + ", ".join(f"{k} ×{v}" for k, v in error_types.most_common()))

    now = datetime.now(timezone.utc)
    run = {"time": now.isoformat(timespec="seconds"), "version": _version(), "model": name, "split": args.split,
           "baseline": args.baseline, "overall": overall, "receipts": results}
    compare_runs(run)
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{now:%Y%m%dT%H%M%SZ}-{run['version']}.json").write_text(json.dumps(run, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(run, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
