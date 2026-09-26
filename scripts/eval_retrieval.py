"""Evaluates the retrieval agent on tests/retrieval_eval.jsonl, then runs the
name typo test.

    python -m scripts.eval_retrieval                  # real embedding model (RETRIEVAL_EMBEDDING_ROUTE)
    python -m scripts.eval_retrieval --no-semantic    # filters + names + boosts only (semantic "failed")
    python -m scripts.eval_retrieval --ids r02,r03 --show

Each row has an extractor output (`filters`, null when the rewriter failed),
the rewriter's `query_en`, the router's `intent` and `k`, and the handles a
good answer should list (`relevant`), plus optional `forbidden` handles
(products with an excluded ingredient). No LLM is called.

Metrics, macro-averaged over rows:
- precision@k: relevant share of the returned list (<= k items).
- recall@k: relevant products returned, over min(|relevant|, k), so a
  k=3 request for 6 relevant sunscreens can still reach 1.0.
- MRR: 1 / rank of the first relevant product.
- forbidden: rows returning a product they must never return.

Targets (retrieval plan, section 9): filters + names + fusion p95 <= 10 ms,
semantic search p95 <= 300 ms; typo test top-1 >= 95%, top-3 >= 99%.
"""

import argparse
import asyncio
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.retrieval import semantic  # noqa: E402
from agents.retrieval.agent import retrieve  # noqa: E402
from agents.retrieval.index_builder import get_index  # noqa: E402
from agents.retrieval.semantic import SemanticScores, semantic_search  # noqa: E402
from config import settings  # noqa: E402
from models.filter_extractor import MetadataFilters  # noqa: E402
from scripts import name_typos  # noqa: E402

EVAL_PATH = ROOT / "tests" / "retrieval_eval.jsonl"
RESULTS_DIR = ROOT / "logs"
TARGET_RETRIEVAL_P95_MS = 10.0
TARGET_SEMANTIC_P95_MS = 300.0


def pct(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q / 100 * (len(s) - 1))))]


def load_rows(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_row(row: dict, handles: list, k: int) -> dict:
    rel = set(row["relevant"])
    hits = [h for h in handles if h in rel]
    first = next((i for i, h in enumerate(handles, 1) if h in rel), None)
    return {
        "precision": len(hits) / len(handles) if handles else 0.0,
        "recall": len(set(hits)) / min(len(rel), k) if rel else 1.0,
        "mrr": 1.0 / first if first else 0.0,
        "forbidden": sorted(set(handles) & set(row.get("forbidden", []))),
    }


async def run_row(row: dict, use_semantic: bool) -> dict:
    filters = MetadataFilters(**row["filters"]) if row["filters"] is not None else None
    sem = await semantic_search(row["query_en"]) if use_semantic else SemanticScores(status="failed", error="--no-semantic")
    res = retrieve(filters, sem, row["query_en"], row["intent"], row["k"])
    return {"id": row["id"], "handles": res.handles, "k": res.k, "relaxed_keys": res.relaxed_keys,
            "name_hits": res.name_hits, "unresolved_names": res.unresolved_names, "fallback": res.fallback,
            "total_candidates": res.total_candidates, "latency_ms": res.meta["latency_ms"], "semantic": res.meta["semantic"],
            "conflicts": {p.handle: p.match["conflict"] for p in res.products if p.match.get("conflict")}}


def report(rows: list, results: dict, args) -> dict:
    by_tag = defaultdict(list)
    scores = []
    for row in rows:
        res = results[row["id"]]
        sc = score_row(row, res["handles"], res["k"])
        res["score"] = sc
        scores.append(sc)
        for tag in row["tags"]:
            by_tag[tag].append(sc)
        bad = sc["recall"] < 1.0 or sc["mrr"] < 1.0 or sc["forbidden"]
        if args.show or bad:
            mark = "ok " if not bad else "-- "
            print(f"{mark}{row['id']} P {sc['precision']:.2f} R {sc['recall']:.2f} MRR {sc['mrr']:.2f}  "
                  f"k={res['k']} C={res['total_candidates']:3d} relaxed={res['relaxed_keys']} {row['query_en'][:60]}")
            if bad or args.show:
                missing = [h for h in row["relevant"] if h not in res["handles"]]
                print(f"      top: {res['handles'][:5]}")
                if missing:
                    print(f"      missing: {missing[:6]}")
                if sc["forbidden"]:
                    print(f"      FORBIDDEN returned: {sc['forbidden']}")

    def avg(key, items):
        return statistics.mean(s[key] for s in items) if items else 0.0

    n = len(rows)
    forbidden_rows = sum(1 for s in scores if s["forbidden"])
    print(f"\n=== retrieval eval, {n} rows (semantic {'on: ' + settings.RETRIEVAL_EMBEDDING_ROUTE if args.semantic else 'off'}) ===")
    print(f"precision@k {avg('precision', scores):.3f}   recall@k {avg('recall', scores):.3f}   MRR {avg('mrr', scores):.3f}   "
          f"rows returning a forbidden product: {forbidden_rows}")
    print("by tag: " + ", ".join(f"{t} R {avg('recall', v):.2f}/MRR {avg('mrr', v):.2f} ({len(v)})"
                                 for t, v in sorted(by_tag.items())))

    own = [results[r["id"]]["latency_ms"]["total"] for r in rows]
    sem_ms = [results[r["id"]]["latency_ms"]["semantic"] for r in rows if results[r["id"]]["semantic"] == "ok"]
    p95 = pct(own, 95)
    print(f"\nfilters + names + fusion: p50 {pct(own, 50):.2f} ms, p95 {p95:.2f} ms, max {max(own):.2f} ms"
          f"   target p95 <= {TARGET_RETRIEVAL_P95_MS:.0f} ms {'PASS' if p95 <= TARGET_RETRIEVAL_P95_MS else 'FAIL'}")
    if sem_ms:
        sp95 = pct(sem_ms, 95)
        print(f"semantic search (uncached query embeddings, {len(sem_ms)} rows): p50 {pct(sem_ms, 50):.1f} ms, p95 {sp95:.1f} ms"
              f"   target p95 <= {TARGET_SEMANTIC_P95_MS:.0f} ms {'PASS' if sp95 <= TARGET_SEMANTIC_P95_MS else 'FAIL'}")
    return {"precision": avg("precision", scores), "recall": avg("recall", scores), "mrr": avg("mrr", scores),
            "forbidden_rows": forbidden_rows, "p95_ms": p95}


def typo_report(seed: int) -> None:
    cases = name_typos.generate(seed=seed)
    stats = name_typos.evaluate(cases)["stats"]
    a = stats["all"]
    top1, top3 = a["top1"] / a["n"], a["top3"] / a["n"]
    print(f"\n=== name typo test, {a['n']} queries (seed {seed}) ===")
    for lang in ("en", "ar"):
        s = stats[lang]
        print(f"{lang}: top-1 {s['top1'] / s['n']:.3f}, top-3 {s['top3'] / s['n']:.3f} (n={s['n']})")
    print(f"all: top-1 {top1:.3f} (target >= {name_typos.TARGET_TOP1}) {'PASS' if top1 >= name_typos.TARGET_TOP1 else 'FAIL'}, "
          f"top-3 {top3:.3f} (target >= {name_typos.TARGET_TOP3}) {'PASS' if top3 >= name_typos.TARGET_TOP3 else 'FAIL'}")
    real = name_typos.real_cases()
    ok = sum(1 for q, top in real.items() if top and top[0] in name_typos.REAL_CASES[q])
    print(f"real misspellings: {ok}/{len(real)} resolved to the right product")


async def run(args) -> int:
    rows = load_rows(Path(args.eval_file) if args.eval_file else EVAL_PATH)
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in wanted]
    get_index()
    if args.semantic:
        await asyncio.to_thread(semantic.warm_up)
    results = {}
    for row in rows:
        results[row["id"]] = await run_row(row, args.semantic)
    summary = report(rows, results, args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"eval_retrieval{'' if args.semantic else '_no_semantic'}.jsonl"
    out.write_text("".join(json.dumps(results[r["id"]], ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"per-row results: {out.relative_to(ROOT)}")
    if not args.skip_typos:
        typo_report(args.seed)
    return 0 if summary["forbidden_rows"] == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the retrieval agent")
    ap.add_argument("--eval-file", default="")
    ap.add_argument("--ids", default="")
    ap.add_argument("--no-semantic", dest="semantic", action="store_false")
    ap.add_argument("--show", action="store_true", help="print every row, not only the imperfect ones")
    ap.add_argument("--skip-typos", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    return asyncio.run(run(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
