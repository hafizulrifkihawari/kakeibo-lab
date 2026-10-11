"""Pick the expense category from the store name and the items.

A port of kakeibo shared/categorizer.ts, so both apps pick the same category.
Keep KNOWN_STORES and KEYWORDS in step with that file. EXTRA_STORES holds stores
that this project found and kakeibo does not list yet.
"""

import re
import unicodedata
from collections import Counter
from typing import Iterable, Optional

I = re.IGNORECASE

# Order matters: the first match wins.
KNOWN_STORES = [
    (re.compile(r"セブン[-ー‐]?イレブン|7[-‐]?ELEVEN|SEVEN[-‐]?ELEVEN", I), "konbini"),
    (re.compile(r"ファミリーマート|FamilyMart|ファミマ", I), "konbini"),
    (re.compile(r"ローソン|LAWSON", I), "konbini"),
    (re.compile(r"ミニストップ|MINISTOP", I), "konbini"),
    (re.compile(r"デイリーヤマザキ"), "konbini"),
    (re.compile(r"セイコーマート"), "konbini"),
    (re.compile(r"NewDays|ニューデイズ", I), "konbini"),
    (re.compile(r"イオン|AEON", I), "groceries"),
    (re.compile(r"まいばすけっと"), "groceries"),
    (re.compile(r"西友|SEIYU", I), "groceries"),
    (re.compile(r"ライフコーポレーション|ライフ(?!スタイル)|LIFE\s*CORPORATION", I), "groceries"),
    (re.compile(r"イトーヨーカ[ドド]ー|Ito[-\s]?Yokado", I), "groceries"),
    (re.compile(r"業務スーパー"), "groceries"),
    (re.compile(r"サミットストア|サミット"), "groceries"),
    (re.compile(r"マルエツ"), "groceries"),
    (re.compile(r"OKストア|オーケー"), "groceries"),
    (re.compile(r"ヤオコー"), "groceries"),
    (re.compile(r"コープ|生協|CO-?OP", I), "groceries"),
    (re.compile(r"マツモトキヨシ|マツキヨ"), "daily"),
    (re.compile(r"ウエルシア"), "daily"),
    (re.compile(r"ツルハ"), "daily"),
    (re.compile(r"スギ薬局"), "daily"),
    (re.compile(r"ココカラファイン"), "daily"),
    (re.compile(r"サンドラッグ"), "daily"),
    (re.compile(r"ダイソー|DAISO", I), "daily"),
    (re.compile(r"セリア|Seria", I), "daily"),
    (re.compile(r"無印良品|MUJI", I), "daily"),
    (re.compile(r"ニトリ|NITORI", I), "daily"),
    (re.compile(r"ドン[・.]?キホーテ|ドンキ"), "daily"),
    (re.compile(r"ユニクロ|UNIQLO", I), "clothing"),
    (re.compile(r"\bGU\b|ジーユー"), "clothing"),
    (re.compile(r"しまむら"), "clothing"),
    (re.compile(r"スターバックス|STARBUCKS", I), "dining"),
    (re.compile(r"ドトール|DOUTOR", I), "dining"),
    (re.compile(r"タリーズ|TULLY", I), "dining"),
    (re.compile(r"マクドナルド|McDonald", I), "dining"),
    (re.compile(r"すき家"), "dining"),
    (re.compile(r"吉野家"), "dining"),
    (re.compile(r"松屋(?!銀座)"), "dining"),
    (re.compile(r"サイゼリヤ"), "dining"),
    (re.compile(r"ガスト"), "dining"),
    (re.compile(r"CoCo壱|ココイチ", I), "dining"),
    (re.compile(r"丸亀製麺"), "dining"),
    (re.compile(r"JR東日本|JR西日本|JR東海|旅客鉄道"), "transport"),
    (re.compile(r"東京メトロ|Tokyo\s*Metro", I), "transport"),
    (re.compile(r"タクシー|TAXI", I), "transport"),
    (re.compile(r"ビックカメラ"), "entertainment"),
    (re.compile(r"ヨドバシ"), "entertainment"),
    (re.compile(r"TSUTAYA|ツタヤ", I), "entertainment"),
]

# Not in kakeibo yet. Add them there too.
EXTRA_STORES = [
    (re.compile(r"GOREMO|ゴーレモ", I), "groceries"),
    (re.compile(r"MATCHAMON|マッチャモン", I), "dining"),
]

# Item keywords, checked only when the store is unknown.
KEYWORDS = [
    (re.compile(r"薬|処方|医院|クリニック|病院|歯科|調剤"), "medical"),
    (re.compile(r"乗車|運賃|切符|乗車券|定期|IC|Suica|PASMO|駐車|ガソリン|給油", I), "transport"),
    (re.compile(r"電気|ガス料金|水道|電話料金|通信料"), "utilities"),
    (re.compile(r"映画|チケット|入場|カラオケ|ゲーム|書籍|書店|雑誌|コミック"), "entertainment"),
    (re.compile(r"シャツ|パンツ|靴|ソックス|ジャケット|衣料"), "clothing"),
    (re.compile(r"ランチ|定食|ラーメン|ドリンク|コーヒー|珈琲|生ビール|お通し|席料|テイクアウト|店内"), "dining"),
    (re.compile(r"洗剤|ティッシュ|トイレット|シャンプー|歯ブラシ|電池|ゴミ袋"), "daily"),
    (re.compile(r"牛乳|野菜|玉子|卵|豆腐|納豆|パン|肉|豚|鶏|牛|魚|米|弁当|おにぎり|惣菜|果物|バナナ|りんご|キャベツ|もやし|ヨーグルト"), "groceries"),
    # Not in kakeibo yet: cafe drinks.
    (re.compile(r"ラテ|カフェ|フラペチーノ|エスプレッソ|カプチーノ"), "dining"),
    (re.compile(r"食品|サントリー|カルビー|明治|森永|ロッテ|グリコ|日清|伊藤園|キリン|アサヒ|コカ[・.]?コーラ|ハウス|味の素|キユーピー|ブルボン|軽減税率"), "groceries"),
]


def _compact(s: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


def categorize(store: Optional[str], item_names: Iterable[str]) -> Optional[str]:
    """The category, or None when neither the store nor the items decide it."""
    s = _compact(store or "")
    if s:
        for pattern, cat in KNOWN_STORES + EXTRA_STORES:
            if pattern.search(s):
                return cat

    # Count hits so that one stray keyword does not win.
    text = unicodedata.normalize("NFKC", "\n".join(item_names))
    scores = Counter()
    for pattern, cat in KEYWORDS:
        scores[cat] += len(pattern.findall(text))
    best = scores.most_common(1)
    return best[0][0] if best and best[0][1] > 0 else None
