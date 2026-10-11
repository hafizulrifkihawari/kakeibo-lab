"""Check a parsed receipt for signs of a bad read, and suggest fixes.

There is no ground truth, so each check looks for an internal contradiction
or a missing value. The score is 100 minus the weight of each failed check.
"""

import re
from datetime import date, timedelta
from typing import List, Optional

from ocr_engine import ParsedReceipt

# Simplified Chinese forms the model sometimes returns in place of the Japanese kanji.
SIMPLIFIED = set("对绿壳卖买费计额预钱备业门东场无时间营发优惠价签现种类")
TAX_RATES = (0.08, 0.10)
QTY_LINE = re.compile(r"^\s*\d+\s*(個|点|コ|本|袋)?\s*[×xX＊*]\s*(単|@)?\s*¥?\d")
MIN_SHORT_SIDE = 1000


def _check(checks: list, name: str, ok: bool, weight: int, detail: str, suggestion: str = "") -> None:
    checks.append({
        "name": name,
        "ok": ok,
        "weight": weight,
        "detail": detail,
        "suggestion": "" if ok else suggestion,
    })


def assess(r: ParsedReceipt, width: Optional[int] = None, height: Optional[int] = None,
           expected_total: Optional[int] = None) -> dict:
    checks: List[dict] = []
    item_sum = sum(i.price for i in r.items)
    charge_sum = sum(c.amount for c in r.charges)
    computed = item_sum + charge_sum

    _check(checks, "Total found", r.total is not None, 25,
           f"total = {r.total}",
           "The model did not find 合計. Make sure the bottom of the receipt is in the photo.")

    if r.total is not None:
        diff = computed - r.total
        _check(checks, "Items and charges add up to the total", diff == 0, 40,
               f"items {item_sum} + charges {charge_sum} = {computed}, total {r.total}, difference {diff:+}",
               _sum_hint(diff, r))

    if expected_total is not None:
        _check(checks, "Total matches the value you entered", r.total == expected_total, 30,
               f"parsed {r.total}, you entered {expected_total}",
               "The total is wrong. Retake the photo with the total line sharp and flat.")

    _check(checks, "Items found", len(r.items) > 0, 20,
           f"{len(r.items)} items",
           "No items were read. Crop the photo to the receipt and retake it in even light.")

    bad_prices = [i.name for i in r.items if i.price <= 0]
    _check(checks, "Item prices are positive", not bad_prices, 5,
           ", ".join(bad_prices) or "all positive",
           "A discount line was read as an item. Add an example of 値引 to the prompt.")

    qty_lines = [i.name for i in r.items if QTY_LINE.search(i.name)]
    _check(checks, "No quantity lines read as items", not qty_lines, 10,
           ", ".join(qty_lines) or "ok",
           "A line such as '2個 × 単120' was read as its own item, so its price is counted twice. "
           "Merge it into the item next to it as qty.")

    bad_chars =sorted({c for i in r.items for c in i.name if c in SIMPLIFIED})
    _check(checks, "Item names use Japanese kanji", not bad_chars, 5,
           "simplified Chinese: " + "".join(bad_chars) if bad_chars else "ok",
           "The model wrote Chinese forms. Normalize them after parsing (kakeibo has a SIMPLIFIED map).")

    _check(checks, "Store found", bool(r.store), 5,
           r.store or "missing",
           "Include the top of the receipt (store name or logo) in the photo.")

    date_ok, date_detail = _date_ok(r.date)
    _check(checks, "Date is valid and recent", date_ok, 10, date_detail,
           "Check the date line. 和暦 (令和) and two-digit years are easy to misread.")

    tax = [c for c in r.charges if c.kind == "tax"]
    if tax and item_sum:
        rate = sum(c.amount for c in tax) / item_sum
        plausible = any(abs(rate - t) < 0.015 for t in TAX_RATES) or 0.08 <= rate <= 0.10
        _check(checks, "Tax is 8% or 10% of the items", plausible, 5,
               f"tax / items = {rate:.1%}",
               "The tax line may be 内税 (already in the prices). Do not add 内税 as a charge.")

    if width and height:
        short = min(width, height)
        _check(checks, "Image resolution", short >= MIN_SHORT_SIDE, 0,
               f"{width} × {height} px",
               f"The short side is {short} px. Use at least {MIN_SHORT_SIDE} px so small kana stay readable.")

    score = max(0, 100 - sum(c["weight"] for c in checks if not c["ok"]))
    return {"score": score, "checks": checks, "suggestions": [c["suggestion"] for c in checks if c["suggestion"]]}


def _sum_hint(diff: int, r: ParsedReceipt) -> str:
    if r.total and diff == round(r.total * 0.1 / 1.1) or r.total and diff == round(r.total * 0.08 / 1.08):
        return "The difference equals the included tax. A 内税 line was probably added as a charge. Remove it."
    if diff < 0:
        return f"{-diff} yen is missing. An item or an 外税 line was skipped, or a qty line was read as one piece."
    return f"{diff} yen too much. A line was read twice, or 小計 was read as an item, or a discount was skipped."


def _date_ok(value: Optional[str]) -> tuple:
    if not value:
        return False, "missing"
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False, f"{value} is not YYYY-MM-DD"
    try:
        d = date.fromisoformat(value)
    except ValueError:
        return False, f"{value} is not a real date"
    today = date.today()
    if d > today:
        return False, f"{value} is in the future"
    if d < today - timedelta(days=366):
        return False, f"{value} is more than a year ago"
    return True, value
