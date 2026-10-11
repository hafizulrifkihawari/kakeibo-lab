"""Parse Japanese receipt images with a local Qwen vision model (Ollama).

The output follows the kakeibo data standard:
- ParsedReceipt (shared/receipt-parser.ts) is what the model returns.
- Expense (shared/types.ts) is what kakeibo saves. to_expense() makes one.
Field names, enums and limits match kakeibo's validateExpense (src/server/fns.ts).
"""

import json
import os
import re
import sys
import time
import uuid
from datetime import date as Date
from typing import List, Literal, Optional, Union

import ollama
from pydantic import BaseModel, Field, ValidationError

MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:7b")


# kakeibo shared/categories.ts
CategoryId = Literal["groceries", "dining", "konbini", "daily", "transport",
                     "medical", "clothing", "entertainment", "utilities", "other"]
# kakeibo shared/products.ts ITEM_KINDS
ItemKind = Literal["produce", "meat_fish", "dairy_eggs", "tofu", "bread_rice", "snacks", "drinks",
                   "alcohol", "frozen", "ready_meals", "seasonings", "household", "personal_care",
                   "medicine", "clothing", "baby", "pet", "stationery", "electronics", "other"]
# kakeibo shared/charges.ts
ChargeKind = Literal["tax", "shipping", "fee", "discount", "other"]


class ReceiptItem(BaseModel):
    name: str = Field(max_length=100, description="Item name in Japanese, exactly as printed")
    price: int = Field(description="Line price in yen: unit price × qty")
    qty: Optional[int] = Field(None, description="Count of a multi-pack line (2–999), or grams / ml when unit is g or ml. Missing means 1")
    unit: Optional[Literal["pc", "g", "ml"]] = Field(None, description="'g' or 'ml' for a line sold by weight or volume. Missing means pieces")
    size: Optional[int] = Field(None, description="Pack size of one piece, such as 300 for '300g', for unit pc only")
    sizeUnit: Optional[Literal["g", "ml"]] = Field(None, description="Unit of size")
    kind: Optional[ItemKind] = Field(None, description="Category of this line")


class Charge(BaseModel):
    kind: ChargeKind
    name: str = Field(max_length=100, description="The label on the receipt, such as 消費税 8%")
    amount: int = Field(description="Yen. Negative for a discount")


class ParsedReceipt(BaseModel):
    store: Optional[str] = Field(description="Store name without 株式会社")
    date: Optional[str] = Field(description="YYYY-MM-DD")
    total: Optional[int] = Field(description="Grand total paid (合計 / 税込合計) in yen")
    categoryId: CategoryId = Field("other", description="Expense category of the whole receipt")
    items: List[ReceiptItem]
    charges: List[Charge] = Field(description="Tax added on top (外税), shipping, fees, discounts")


def clean_item(i: ReceiptItem) -> ReceiptItem:
    """Apply kakeibo's qtyAndUnit and packSize rules, so the item saves unchanged."""
    d = {"name": i.name.strip()[:100], "price": i.price, "kind": i.kind}
    q = i.qty
    if i.unit in ("g", "ml") and q is not None and 1 <= q <= 100_000:
        d.update(qty=q, unit=i.unit)
    else:
        if q is not None and 1 < q < 1000:
            d["qty"] = q
        if i.size == 0:
            d["size"] = 0
        elif i.size is not None and 1 <= i.size <= 100_000 and i.sizeUnit in ("g", "ml"):
            d.update(size=i.size, sizeUnit=i.sizeUnit)
    return ReceiptItem(**d)


def clean(r: ParsedReceipt) -> ParsedReceipt:
    return r.model_copy(update={
        "store": (r.store or "").strip()[:100] or None,
        "items": [clean_item(i) for i in r.items if i.name.strip()][:200],
        "charges": [c.model_copy(update={"name": c.name.strip()[:100]}) for c in r.charges][:20],
    })


# Charge lines that only repeat other lines: subtotals, tax bases, included tax, and payment.
NOT_A_CHARGE = re.compile(r"小\s*計|合\s*計|対象|内税|内消費税|税合計|お?預|お?釣|釣銭|クレジット|現金|電子マネー|カード|支払|ポイント")
QTY_LINE = re.compile(r"^\s*(\d+)\s*(?:個|点|コ|本|袋)?\s*[×xX＊*]")
CODE_PREFIX = re.compile(r"^\d+\s*※\s*")


def fix(r: ParsedReceipt) -> ParsedReceipt:
    """Correct the errors the model makes most often. Run after clean()."""
    # Summary and payment lines are not charges. A charge of -total is the payment line.
    charges = [c for c in r.charges
               if not NOT_A_CHARGE.search(c.name) and not (r.total and c.amount == -r.total)]

    # Remove store codes such as "001029※" from item names.
    items = [i.model_copy(update={"name": CODE_PREFIX.sub("", i.name) or i.name}) for i in r.items]

    # A quantity line such as "2個 x 0129" belongs to the item before it.
    merged: List[ReceiptItem] = []
    for idx, i in enumerate(items):
        m = QTY_LINE.match(i.name)
        target = merged[-1] if merged else (items[idx + 1] if idx + 1 < len(items) else None)
        if m and target is not None:
            target.qty = int(m.group(1))
            target.price = i.price
            continue
        merged.append(i)

    # The model sometimes gives the unit price of a qty > 1 line. Use price × qty when that fixes the sum.
    if r.total is not None:
        diff = r.total - sum(i.price for i in merged) - sum(c.amount for c in charges)
        for i in merged:
            if diff and i.qty and i.qty > 1 and (i.unit or "pc") == "pc" and i.price * (i.qty - 1) == diff:
                i.price *= i.qty
                diff = 0

    return r.model_copy(update={"items": merged, "charges": charges})


def to_expense(r: ParsedReceipt, raw_text: str = "") -> dict:
    """The kakeibo Expense (shared/types.ts) for a parsed receipt."""
    amount = r.total if r.total is not None else sum(i.price for i in r.items) + sum(c.amount for c in r.charges)
    return {
        "id": str(uuid.uuid4()),
        "date": r.date or Date.today().isoformat(),
        "amount": amount,
        "store": r.store or "",
        "categoryId": r.categoryId,
        "note": "",
        "ocrEngine": "vision",
        "rawText": raw_text,
        "items": [i.model_dump(exclude_none=True) for i in r.items],
        "charges": [c.model_dump() for c in r.charges],
        "updatedAt": int(time.time() * 1000),
    }


SYSTEM_PROMPT = """You read Japanese receipts and return JSON only.
Rules:
1. total is the grand total paid (合計, 税込合計, お会計). Not 小計, お預り, or お釣り.
2. Keep item names in Japanese exactly as printed. Do not translate.
3. Yen has no decimals. All amounts are integers.
4. A quantity line such as "2個 × 単158" is not an item. It belongs to the item name on the line next to it.
   Merge them: one item with that name, qty 2, and the line price 316.
5. "298g × @198/100g" is one item with qty 298, unit "g", and the printed line price.
6. 外税 (tax added on top) is a charge with kind "tax".
7. 内税 (tax already in the prices) is not a charge. Do not add it.
8. 値引 or 割引 is a charge with kind "discount" and a negative amount.
9. Dates: "2026年10月11日", "26.10.11", and 令和8年10月11日 all become "2026-10-11".
10. Use null when a value is not on the receipt.
11. size is the pack size printed in the name ("牛乳 1000ml" → size 1000, sizeUnit "ml"). Keep it in the name too.
12. kind is the category of each item: produce, meat_fish, dairy_eggs, tofu, bread_rice, snacks, drinks,
    alcohol, frozen, ready_meals, seasonings, household, personal_care, medicine, clothing, baby, pet,
    stationery, electronics, or other.
13. categoryId is the category of the whole receipt: groceries (supermarket), dining (restaurant, cafe),
    konbini (convenience store), daily, transport, medical, clothing, entertainment, utilities, or other."""


def parse_receipt(image: Union[str, bytes]) -> ParsedReceipt:
    """Parse one receipt. image is a file path or the raw image bytes."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Read this receipt.", "images": [image]},
        ],
        format=ParsedReceipt.model_json_schema(),
        options={"temperature": 0, "num_ctx": 8192},
    )
    return fix(clean(ParsedReceipt.model_validate_json(response.message.content)))


def check_total(receipt: ParsedReceipt, image_path: str) -> None:
    if receipt.total is None:
        return
    computed = sum(i.price for i in receipt.items) + sum(c.amount for c in receipt.charges)
    if computed != receipt.total:
        print(f"warning: {image_path}: items + charges = {computed}, total = {receipt.total}", file=sys.stderr)


def main(args: List[str]) -> int:
    as_expense = "--expense" in args
    paths = [a for a in args if a != "--expense"]
    if not paths:
        print("usage: python ocr_engine.py [--expense] IMAGE [IMAGE ...]", file=sys.stderr)
        return 2
    results, failed = [], False
    for path in paths:
        try:
            receipt = parse_receipt(path)
        except (ValidationError, ollama.ResponseError, OSError) as e:
            print(f"error: {path}: {e}", file=sys.stderr)
            failed = True
            continue
        check_total(receipt, path)
        results.append(to_expense(receipt) if as_expense else receipt.model_dump(exclude_none=True))
    out = results[0] if len(paths) == 1 and results else results
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
