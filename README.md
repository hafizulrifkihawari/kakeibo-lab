# kakeibo-lab

A local lab for the receipt reading of [kakeibo](https://github.com/hafizulrifkihawari/kakeibo).

- Read Japanese receipt photos offline with Qwen 2.5 VL on Ollama. The JSON output matches `ParsedReceipt` in kakeibo.
- Collect the receipts you scan in kakeibo, and review them into labeled receipts.
- Measure each change to the prompt, the rules, or the model against those labels. Keep it only if the score goes up.

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
  "items": [{ "name": "おにぎり 鮭", "reading": "おにぎり しゃけ", "en": "Salmon rice ball", "price": 160, "kind": "ready_meals" }],
  "charges": []
}
```

Each item has its reading in hiragana (`reading`, furigana) and a short English translation (`en`). The model writes the translation in the same call. The reading comes from the pykakasi dictionary after the call, because the model often copies the name instead of reading it. Only kanji get a reading in hiragana. Katakana, Latin letters, and numbers stay as printed. A name with no kanji has no reading. Saved labels and the kakeibo `Expense` leave them out: kakeibo makes its own (`item_gloss`).

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
The same page reviews the receipts that come from kakeibo. See [Review](#review).

## Improve the parser

The model does not learn from use. Use this loop to make the parser better and to prove each change.

1. Get labeled receipts. A labeled receipt is an image and a `.json` file with the correct answer, in `labels/` (gitignored). There are two ways to get them:
   - **From kakeibo (automatic):** see [Learn from kakeibo scans](#learn-from-kakeibo-scans).
   - **By hand:** parse a receipt in the web UI. Fix the JSON in **Correct and save** so it matches the receipt exactly. Then click **Save as correct**.
2. Measure the parser:

   ```bash
   .venv/bin/python eval.py                  # the dev split
   .venv/bin/python eval.py --split test     # the test split: check it before you keep a change
   .venv/bin/python eval.py --baseline prod  # kakeibo's own scan on the same receipts
   ```

   The output gives a score for each field (store, date, total, category, items, charges, kind), one overall score, and the most common errors.
3. Change one thing: the prompt (`SYSTEM_PROMPT`), the `fix()` rules, or the model (`OLLAMA_MODEL`). Run `eval.py` again.
4. Keep the change only if the dev score goes up and the test score does not go down.

How `eval.py` works:

| Part | What it does |
|---|---|
| Split | 1 in 5 receipts is in the **test** split, the others are in **dev**. The file name fixes the split, so it never changes. Tune on dev. Use test only to check a change. A fine-tune later trains on dev only. |
| Cache | The model's raw answers are kept in `.cache/<model>/<prompt>/`. A change to `clean()`, `fix()`, or the category rules runs in seconds. A change to the prompt, the schema, or the model reads every image again (about 20 s to 3 min per image). |
| History | Each run is saved in `runs/`. The output shows the change from the last run of the same kind, for each field and for each receipt that got worse. |
| Baseline | `--baseline prod` scores the first reading of kakeibo's scanner (Google Vision and the regex parser). Only receipts reviewed from the kakeibo inbox have one. kakeibo's scan does not read item kinds, so `kind` is not scored. |

Milestones:

- About 30 labeled receipts: the comparison of the model with kakeibo's scanner starts to mean something.
- 150–300 labeled receipts: you can fine-tune a LoRA adapter for Qwen2.5-VL on the dev split, on your Mac. Training is a manual step that runs for hours. Start it in the evening or on a weekend. The daily sync job never trains.

Planned work, including quick fixes and a plan to distill a model that you own, is in [TODO.md](TODO.md).

## Learn from kakeibo scans

[kakeibo](https://github.com/hafizulrifkihawari/kakeibo) can keep the photo of each receipt that you scan. This tool downloads those photos with the expense that you saved. You review each one, and it becomes a labeled receipt.

```
phone: scan in kakeibo, fix, save ──► kakeibo D1: receipt_scans (photo + first reading)
Mac:   sync.py ──► labels/inbox/ ──► review in the web UI ──► labels/ ──► eval.py
```

It is free. The photo waits in kakeibo's D1 database only until the next sync, which deletes it there. The model runs on your Mac.

### Requirements

- macOS, with Ollama and `qwen2.5vl:7b` (see [Setup](#setup)).
- Node.js (for `npx wrangler`).
- Your own kakeibo deployment. Follow its README: "Deploy (free tier)", then "Collect scans for a local model". Scan collection is off by default. Turn it on for your account with `npm run admin -- collect you@example.com on --remote`.
- A clone of kakeibo on this Mac, and `npx wrangler login` with the Cloudflare account of your deployment.

### Configuration

Set these in the environment or in a `.env` file in this folder (gitignored):

| Variable | Default | What it is |
|---|---|---|
| `KAKEIBO_DIR` | `../kakeibo` | The kakeibo clone. A relative path starts from this folder. |
| `KAKEIBO_D1` | `jp-expense` | The `database_name` in kakeibo's `wrangler.jsonc`. |
| `OLLAMA_MODEL` | `qwen2.5vl:7b` | The model. |

### First run

```bash
.venv/bin/python sync.py --dry-run    # show what waits in kakeibo. Changes nothing.
.venv/bin/python sync.py --predict    # download, then let the model read each new photo
```

For each scan, `sync.py` writes three files to `labels/inbox/`: the photo, the expense you saved (`<id>.json`), and kakeibo's first reading (`<id>.prod.json`). Then it deletes the scan in kakeibo. It skips a scan whose expense is not saved yet (it waits in the phone's offline queue). It deletes a scan whose expense you deleted. `--keep` keeps the scans in kakeibo. `--local` uses kakeibo's local development database.

### Daily job

```bash
.venv/bin/python sync.py --install     # run now, every day at 19:00, and at login
.venv/bin/python sync.py --uninstall   # stop it
launchctl print gui/$(id -u)/com.kakeibo-lab.sync | grep -E 'state|last exit'
```

- The job always downloads new scans. That takes a few seconds.
- The model reads new photos only from 18:00 to 06:00, only on AC power, and only when Ollama runs. So a login run in the morning stays light. Set the Ollama app to open at login.
- The job unloads the model after each photo (`keep_alive: 0`), so its memory (6–8 GB) is free again at once.
- If the Mac sleeps at 19:00, the job runs when it wakes. If the Mac is off, the job runs at the next login.
- Each run adds a line to `logs/sync.log`. A failed sync (for example, an expired wrangler login) shows a macOS notification.
- A job that never starts cannot send a notification. So the web UI shows the last sync time, in red after 48 hours.
- `--install` is safe to run again. Run it again after you move this folder or change the Node.js version.

**Important: a folder in ~/Documents, ~/Desktop, or ~/Downloads.** macOS does not let a background job read these folders. The job then stops before it starts, and `logs/launchd.log` shows `Operation not permitted`. Do one of these:

1. Move this folder and kakeibo to a folder such as `~/Developer`. Then run `--install` again. This is the reliable fix.
2. Give Full Disk Access to the Python binary that `--install` prints (System Settings > Privacy & Security > Full Disk Access). A Homebrew Python upgrade changes that path, so do this again after each upgrade.

### Review

```bash
.venv/bin/python app.py    # then open http://localhost:8000
```

The **Inbox from kakeibo** card shows how many receipts wait and when the last sync ran. Click **Review next**:

- The photo is on the left. On the right, a table compares the label (the expense you saved), the model's reading, and kakeibo's first reading. A red cell does not agree with the label: check it on the photo.
- The item table marks each label item: ✓ the same name and price, ≈ the same price and another name, ✗ missing. Items that only the model or kakeibo found are listed below it.
- Fix the label JSON. **Use the model's reading** copies the model's answer into the label as a start. **Run the model** reads the photo now, if the daily job did not.
- **Accept** saves the label to `labels/<id>.json` and keeps kakeibo's first reading in `labels/prod/` for `--baseline prod`. **Discard** removes a bad photo.

You often fix only the total and the category in kakeibo. So check every item name and price before you accept.

The **Labeled receipts** card lists every saved label with its date, store, total, split, and whether it has a kakeibo scan. Click one to open it in the same review form. **Save changes** overwrites its label. **Run the model** reads it again. **Delete label** deletes the label, its photo, and the kakeibo scan, for example a receipt that you saved twice.

### Troubleshooting

| Problem | Fix |
|---|---|
| `wrangler is not logged in` | Run `npx wrangler login` in the kakeibo folder. |
| `logs/launchd.log` shows `Operation not permitted` | See the note about ~/Documents above. |
| The job cannot find `npx` | You changed the Node.js version. Run `--install` again. |
| `predicted=0 (model skipped: Ollama is not running)` | Open the Ollama app. Set it to open at login. |
| The inbox stays empty | Scan collection is off for your account (kakeibo: `npm run admin -- collect you@example.com on --remote`). The app sees the change within 5 minutes. Or the receipt was not scanned from a photo, or the phone was offline when you saved it. |
| `no such table: receipt_scans` | Apply kakeibo's migrations: `npx wrangler d1 migrations apply jp-expense --remote`. |

### Privacy

`labels/`, `.cache/`, `runs/`, and `logs/` hold photos and data of your purchases. They are gitignored. Never commit them.
