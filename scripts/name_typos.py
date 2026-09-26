"""Misspelled product names for the name-search test (retrieval plan, section 10).

    python -m scripts.name_typos            # top-1 / top-3 accuracy per language, plus the misses
    python -m scripts.name_typos --seed 7

For every product name, English and Arabic, five variants are generated,
deterministically from the seed:

1. one letter deleted
2. two adjacent letters swapped
3. one letter replaced
4. two edits (any mix of the above)
5. the brand word dropped. This variant is skipped when nothing is left, or
   when what is left equals another product's name without its brand
   ("Movelex Nano Spray" and "Methytral Nano Spray" are both "nano spray").

Only letters in words of 3+ letters are edited. A digit typo changes the
meaning ("Urea 15%" -> "Urea 5%"), and a 1-2 letter word ("c" in "vitamin c")
has nothing left to match.

A query counts as correct when the top hit is the product, or a product with
the same normalized name (Movelex Cream 120gm vs 50gm cannot be told apart
by name). Targets: top-1 >= 95%, top-3 >= 99%.
"""

import argparse
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.filter_extractor.catalog import METADATA_PATH, _load_json  # noqa: E402
from agents.retrieval.index_builder import RetrievalIndex, get_index  # noqa: E402
from agents.retrieval.name_search import match_name, normalize_name  # noqa: E402

TARGET_TOP1 = 0.95
TARGET_TOP3 = 0.99
EN_LETTERS = "abcdefghijklmnopqrstuvwxyz"
AR_LETTERS = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
MIN_WORD = 3

# Real misspellings from the plan: query -> acceptable top-1 handles.
REAL_CASES = {
    "capixi dray foom": {"capixy-dry-foam-120-ml"},
    "soralon cica": {"soralonecica", "soralone-cica-roll-on-50-ml", "soralone-cica-lip-balm-5ml", "soralonencs100ml"},
    "vaccation niacinamid": {"vacation-niacinamide-10-zinc-pca-serum-30-ml"},
    "كابكسي فوم": {"capixy-dry-foam-120-ml"},
    "ميثترال سبراي": {"methytral-nano-spray-100ml"},
}


@dataclass
class TypoCase:
    query: str
    handle: str
    lang: str
    kind: str        # delete | swap | replace | two_edits | no_brand
    name: str        # the normalized name it was made from


def _letter_positions(text: str) -> List[int]:
    out, i = [], 0
    for word in text.split(" "):
        if sum(ch.isalpha() for ch in word) >= MIN_WORD:
            out.extend(i + j for j, ch in enumerate(word) if ch.isalpha())
        i += len(word) + 1
    return out


def _edit(text: str, kind: str, rng: random.Random, letters: str) -> Optional[str]:
    positions = _letter_positions(text)
    if not positions:
        return None
    if kind == "delete":
        i = rng.choice(positions)
        return text[:i] + text[i + 1:]
    if kind == "replace":
        i = rng.choice(positions)
        return text[:i] + rng.choice([c for c in letters if c != text[i]]) + text[i + 1:]
    if kind == "swap":
        pairs = [i for i in positions if i + 1 in positions and text[i] != text[i + 1]]
        if not pairs:
            return None
        i = rng.choice(pairs)
        return text[:i] + text[i + 1] + text[i] + text[i + 2:]
    raise ValueError(kind)


def _brand_phrases(index: RetrievalIndex, lang: str) -> Dict[str, List[str]]:
    """brand -> its normalized spellings in one language, longest first."""
    meta = _load_json(METADATA_PATH)
    brands = {p["brand"] for p in index.products.values()}
    out = {}
    for b in brands:
        forms = meta.get("arabic_hints", {}).get("brand", {}).get(b, []) if lang == "ar" else [b] + meta["brand_aliases"].get(b, [])
        out[b] = sorted({normalize_name(f, lang) for f in forms if normalize_name(f, lang)}, key=len, reverse=True)
    return out


def drop_brand(name: str, phrases: Sequence[str]) -> str:
    words = f" {name} "
    for ph in phrases:
        words = words.replace(f" {ph} ", " ")
    return " ".join(words.split())


def generate(index: Optional[RetrievalIndex] = None, seed: int = 13) -> List[TypoCase]:
    index = index or get_index()
    rng = random.Random(seed)
    cases: List[TypoCase] = []
    for lang, name_index, letters in (("en", index.names_en, EN_LETTERS), ("ar", index.names_ar, AR_LETTERS)):
        phrases = _brand_phrases(index, lang)
        no_brand = {h: drop_brand(t, phrases[index.products[h]["brand"]]) for h, t in zip(name_index.handles, name_index.texts)}
        taken = Counter(no_brand.values())
        for handle, text in zip(name_index.handles, name_index.texts):
            for kind in ("delete", "swap", "replace"):
                q = _edit(text, kind, rng, letters)
                if q and q != text:
                    cases.append(TypoCase(q, handle, lang, kind, text))
            q = _edit(text, rng.choice(("delete", "swap", "replace")), rng, letters)
            q = q and _edit(q, rng.choice(("delete", "swap", "replace")), rng, letters)
            if q and q != text:
                cases.append(TypoCase(q, handle, lang, "two_edits", text))
            nb = no_brand[handle]
            if nb and nb != text and taken[nb] == 1:
                cases.append(TypoCase(nb, handle, lang, "no_brand", text))
    return cases


def evaluate(cases: Sequence[TypoCase], index: Optional[RetrievalIndex] = None) -> dict:
    index = index or get_index()
    same_name = {}
    for name_index in (index.names_en, index.names_ar):
        by_text = defaultdict(set)
        for h, t in zip(name_index.handles, name_index.texts):
            by_text[t].add(h)
        same_name[name_index.lang] = {h: by_text[t] for h, t in zip(name_index.handles, name_index.texts)}
    stats = defaultdict(Counter)
    misses = []
    for c in cases:
        hits, _ = match_name(c.query, index.names_ar if c.lang == "ar" else index.names_en)
        ok = same_name[c.lang][c.handle]
        top = [h.handle for h in hits]
        rank = next((i for i, h in enumerate(top) if h in ok), None)
        for key in (c.lang, "all", f"{c.lang}:{c.kind}"):
            stats[key]["n"] += 1
            stats[key]["top1"] += rank == 0
            stats[key]["top3"] += rank is not None and rank < 3
        if rank != 0:
            misses.append((c, top[:3], rank))
    return {"stats": stats, "misses": misses}


def real_cases(index: Optional[RetrievalIndex] = None) -> Dict[str, List[str]]:
    index = index or get_index()
    out = {}
    for q in REAL_CASES:
        hits, _ = match_name(q, index.names_ar if any("؀" <= ch <= "ۿ" for ch in q) else index.names_en)
        out[q] = [h.handle for h in hits]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Name-search typo test")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--show-misses", type=int, default=30)
    args = ap.parse_args(argv)
    cases = generate(seed=args.seed)
    res = evaluate(cases)
    stats = res["stats"]
    print(f"=== name typo test, {len(cases)} queries (seed {args.seed}) ===")
    for key in sorted(stats, key=lambda k: (":" in k, k)):
        s = stats[key]
        print(f"{key:16s} n={s['n']:4d}  top-1 {s['top1'] / s['n']:.3f}  top-3 {s['top3'] / s['n']:.3f}")
    a = stats["all"]
    top1, top3 = a["top1"] / a["n"], a["top3"] / a["n"]
    print(f"\ntop-1 {top1:.3f} (target >= {TARGET_TOP1}) {'PASS' if top1 >= TARGET_TOP1 else 'FAIL'}; "
          f"top-3 {top3:.3f} (target >= {TARGET_TOP3}) {'PASS' if top3 >= TARGET_TOP3 else 'FAIL'}")
    print("\nreal cases:")
    for q, top in real_cases().items():
        verdict = "ok " if top and top[0] in REAL_CASES[q] else "ERR"
        print(f"  {verdict} {q!r:28s} -> {top[:4]}")
    if res["misses"] and args.show_misses:
        print(f"\nmisses (first {args.show_misses}):")
        for c, top, rank in res["misses"][: args.show_misses]:
            print(f"  [{c.lang}:{c.kind}] {c.query!r} (from {c.name!r}) -> rank {rank}, top {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
