# receipt-parser-jp

Parse Japanese receipt images offline with Qwen 2.5 VL on Ollama.
The JSON output matches `ParsedReceipt` in the kakeibo app.

## Setup

```bash
ollama pull qwen2.5vl:7b
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```bash
.venv/bin/python ocr_engine.py receipt.jpg
```

Output:

```json
{
  "store": "ローソン",
  "date": "2026-10-11",
  "total": 160,
  "categoryId": "konbini",
  "items": [{ "name": "おにぎり 鮭", "price": 160, "kind": "ready_meals" }],
  "charges": []
}
```

Add `--expense` to print a kakeibo `Expense` (`shared/types.ts`) that kakeibo can save as is.
It has `id`, `amount`, `categoryId`, `ocrEngine`, `updatedAt`, and the items and charges.
The item fields (`qty`, `unit`, `size`, `sizeUnit`, `kind`) follow the limits in kakeibo's `validateExpense`.

Set `OLLAMA_MODEL` to use a different model tag.
A warning goes to stderr when the items and charges do not add up to the total.

## Web UI

```bash
.venv/bin/python app.py
```

Open http://localhost:8000 and drop a receipt image on the page.
The page shows the parsed receipt, a confidence score out of 100, each check, and suggestions.
Enter the real total to check the parsed total against it.
The score comes from checks that look for contradictions and missing values in the result (`assess.py`). It is not compared with a hand-checked answer.
