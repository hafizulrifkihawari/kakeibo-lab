# To do

Work for later. Measure each change with `eval.py` and keep it only if the score goes up (see README, "Improve the parser").

## Quick fixes

Errors that the eval shows now (3 labels, October 2026):

- [ ] **Store name.** The model drops the chain name: `GOREMO 千本中立売店` → `千本中立売店`, `MATCHAMON 京都二条城東店` → `京都二条城東店`. Add a prompt rule: keep the chain or brand name that is printed on the receipt.
- [ ] **Item count as a discount.** The model reads `買上点数 -7` as a discount. Add `点数` to `NOT_A_CHARGE` in `ocr_engine.py`.
- [ ] **Item kind (69%).** Look up item names in kakeibo's grocery catalog (`shared/grocery-catalog.ts`), as `categorizer.py` does for the category. Use the model's kind only as a fallback.
- [ ] **Wrong readings.** pykakasi reads some compound words incorrectly: 鶏もも → にわとりもも (とりもも), 四川風 → しせんかぜ (しせんふう), 堅あげ → けんあげ (かたあげ). Add a small list of fixes in `kana_reading()`.
- [ ] **Remove `reading` from the model schema.** pykakasi makes the reading now, so the model does not need to write it. This makes the model faster. The cache key changes, so the model reads all photos again.
- [ ] **Better English.** The model's translations are rough ("Calbee Kangaage Po"). Try a text-only call with kakeibo's gloss prompt (`src/server/gloss.ts`), which also completes cut-off names ("堅あげポ" → "堅あげポテト"). Needs a text model, for example `qwen2.5:7b` (about 5 GB).
- [ ] **Port findings to kakeibo.** kakeibo's phone app cannot use the model on this Mac. Use `eval.py --baseline prod` errors to improve kakeibo's regex parser, store list, and category rules.

## Later: distill a model that you own

Goal: a model whose weights you own and can train again whenever you have more labels.

### Design

1. A strong **teacher** model reads each receipt photo and writes a draft label.
2. You review the draft in the review page. The reviewed label is the gold label. Do not train on teacher drafts that nobody reviewed: the student then learns the teacher's mistakes.
3. A small **student** model (Qwen2.5-VL 3B or 7B) gets a LoRA fine-tune on the gold labels of the **dev** split only.
4. The **test** split decides: the student against the current model, the teacher, and kakeibo's scan (`--baseline prod`).
5. The student runs on this Mac. Add it as a second backend in `ocr_engine.py`, next to Ollama.

### Decisions to make

| Decision | Options |
|---|---|
| Teacher | A cloud model (Claude, Gemini, GPT): best drafts, small cost per receipt, photos go to the provider. Check the provider's terms on training other models with its outputs. Or a bigger local Qwen2.5-VL (32B): free and private, slow, needs about 20+ GB of RAM. Or no teacher: train on your labels only. |
| Student size | 3B: about 2× faster, easier to train on a Mac. 7B: more accurate, needs more memory. |
| Mac memory | Check with `sysctl -n hw.memsize`. A 7B LoRA probably needs 32 GB or more. |
| More photos | Import old receipt photos from the phone gallery (`PXL_*.jpg`) in bulk. The teacher drafts them, and you review them in the same page. |

### Check before you build

- [ ] Does `mlx-vlm` support LoRA training for Qwen2.5-VL on Apple Silicon?
- [ ] Can a fine-tuned Qwen2.5-VL adapter be imported into Ollama (with its vision part)? If not, serve the student with `mlx-vlm` directly.
- [ ] The license of Qwen2.5-VL 3B. It may be stricter than the Apache 2.0 license of 7B.

### When to start

- About 100 gold labels: the test split (1 in 5) becomes large enough to judge a fine-tune.
- 150–300 gold labels: a LoRA fine-tune is worth trying. A teacher and a bulk import of old photos get you there faster.

### Steps

1. `teacher.py`: draft labels for the inbox and for a folder of old photos.
2. Bulk import: copy old photos into `labels/inbox/` with teacher drafts as the label.
3. `train.py`: LoRA fine-tune on the dev split, with a fixed seed. Save the adapter with its training data list and eval score.
4. A second backend in `ocr_engine.py` (`OCR_BACKEND=mlx`), so `eval.py` and the web UI can use the student.
5. Compare on the test split. Keep the student only if it wins.
