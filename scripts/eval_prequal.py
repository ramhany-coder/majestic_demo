"""Evaluates the pre-qualification stage (rewriter + router) on tests/prequal_eval.jsonl.

    python -m scripts.eval_prequal                        # live prequal, extractor stage with the offline rules
    python -m scripts.eval_prequal --extractor llm        # live extractor too (≈7k tokens per row on Groq)
    python -m scripts.eval_prequal --mode rescore         # re-score saved outputs (no API calls)
    python -m scripts.eval_prequal --limit 10 --ids c01,c15

Each row is one conversation turn: history, last products and the new
message, with the expected rewrite meaning and routing labels.

- Router metrics: accuracy of route, needs_retrieval, intent, persona and
  k (the count the user asked for; a row without expected.k expects null);
  plus the rewriter's language and is_follow_up. Rows list every
  acceptable intent / persona where the label is a judgment call.
- Rewrite check: the rewrite must contain one alternative from every
  `must_include` group and none of the `must_not_include` terms
  (case-insensitive substrings). A topic change row lists the old topic's
  words under must_not_include.
- End-to-end rewrite check: the metadata extractor runs on query_en and its
  filters are scored against the row's expected filters with the extractor
  eval's rules (optional values neither rewarded nor penalized), per key F1.
- Latency: prequal p50 / p95, and the full products_only path (prequal +
  extractor || semantic search + retrieval) p95 over rows routed
  products_only. The full-path number is only meaningful with --extractor llm.

Targets (plan section 9/8): route >= 0.95, needs_retrieval >= 0.97, e2e
micro F1 >= 0.85, prequal p50 <= 700 ms / p95 <= 1200 ms, products_only
path p95 <= 3000 ms.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.filter_extractor.agent import extract_filters  # noqa: E402
from agents.orchestrator.orchestrator import warm_up  # noqa: E402
from agents.prequal.agent import get_cache, prequalify  # noqa: E402
from agents.prequal.schemas import validate_route  # noqa: E402
from agents.prequal.session_context import SessionContext  # noqa: E402
from agents.retrieval.agent import retrieve  # noqa: E402
from agents.retrieval.semantic import semantic_search  # noqa: E402
from models.filter_extractor import MetadataFilters  # noqa: E402
from scripts.eval_extractor import ALL_KEYS, TokenPacer, pct, predicted, rules_only, score_row  # noqa: E402

EVAL_PATH = ROOT / "tests" / "prequal_eval.jsonl"
RESULTS_DIR = ROOT / "logs"
LABELS = ("route", "needs_retrieval", "intent", "persona", "k", "language", "is_follow_up")
ROUTER_KEYS = ("route", "needs_retrieval", "intent", "persona", "k")
TARGETS = {"route": 0.95, "needs_retrieval": 0.97}
EST_PREQUAL_TOKENS = 1150
EST_EXTRACTOR_TOKENS = 6500


def rewrite_check(query_en: str, expected: dict) -> list:
    """Returns the failed checks (empty list = pass)."""
    text = query_en.lower()
    failed = [f"missing {group}" for group in expected["must_include"]
              if not any(alt.lower() in text for alt in group)]
    failed += [f"has '{term}'" for term in expected["must_not_include"] if term.lower() in text]
    return failed


def label_ok(key: str, pred: dict, expected: dict) -> bool:
    exp = expected.get(key)          # only k may be missing: null expected
    return pred.get(key) in exp if isinstance(exp, list) else pred.get(key) == exp


def router_k(pq) -> object:
    """The router's own k (None when it stated no count), not the defaulted pq.k."""
    return pq.k if pq.meta.get("k_source") == "router" else None


def ctx_for(row: dict) -> SessionContext:
    return SessionContext(history=list(row["history"]), last_products=list(row["last_products"]))


async def run_row(row: dict, args, pacer: TokenPacer) -> dict:
    est = EST_PREQUAL_TOKENS + (EST_EXTRACTOR_TOKENS if args.extractor == "llm" else 0)
    await pacer.wait(est)
    pq = await prequalify(row["query"], ctx_for(row), use_cache=False)
    used = sum(pq.meta.get("prompt_tokens", {}).values())
    out = {
        "id": row["id"], "query": row["query"],
        "pred": {k: router_k(pq) if k == "k" else getattr(pq, k) for k in LABELS},
        "query_en": pq.query_en, "skip_metadata_filters": pq.skip_metadata_filters,
        "prequal_ms": pq.meta["latency_ms"]["total"], "call_ms": {k: v for k, v in pq.meta["latency_ms"].items() if k != "total"},
        "status": pq.meta["status"], "prompt_tokens": pq.meta.get("prompt_tokens", {}), "notes": pq.meta.get("notes", []),
    }
    if row.get("filters") is not None and args.extractor != "off":
        # Same shape as the orchestrator: extractor || semantic search, then retrieval.
        t = time.perf_counter()
        if pq.skip_metadata_filters:
            f, sem = None, await semantic_search(pq.query_en)
        elif args.extractor == "llm":
            f, sem = await asyncio.gather(extract_filters(pq.query_en, use_cache=False), semantic_search(pq.query_en))
            used += sum(f.meta.prompt_tokens.values())
        else:
            f, sem = rules_only(pq.query_en, []), await semantic_search(pq.query_en)
        out["extractor_ms"] = round((time.perf_counter() - t) * 1000, 1)
        found = retrieve(f, sem, pq.query_en, pq.intent, pq.k)
        out["retrieval_ms"] = found.meta["latency_ms"]["total"]
        out["semantic"] = sem.status
        out["retrieved"] = found.handles[:5]
        f = f or MetadataFilters()
        out["filters"] = predicted(f)
        out["extractor_calls"] = f.meta.calls
    if used:
        pacer.record(used)
    return out


def score(rows: list, results: dict, args) -> None:
    label_hits = Counter()
    rewrite_pass, tag_total, tag_pass = 0, Counter(), Counter()
    totals = {k: Counter() for k in ALL_KEYS}
    prequal_ms, path_ms, call_ms = [], [], defaultdict(list)
    statuses, tokens = Counter(), defaultdict(list)
    n_e2e = 0
    for row in rows:
        res = results[row["id"]]
        exp = row["expected"]
        errs = []
        for k in LABELS:
            if label_ok(k, res["pred"], exp):
                label_hits[k] += 1
            else:
                errs.append(f"{k}={res['pred'][k]!r} (want {exp[k]})")
        rw_fail = rewrite_check(res["query_en"], exp)
        rewrite_pass += not rw_fail
        for tag in row["tags"]:
            tag_total[tag] += 1
            tag_pass[tag] += not rw_fail
        if rw_fail:
            errs.append(f"rewrite {rw_fail}")
        if "filters" in res and row.get("filters") is not None:
            n_e2e += 1
            sc = score_row({"expected": row["filters"], "optional": row["optional_filters"]}, res["filters"])
            for k, v in sc.items():
                totals[k]["tp"] += v["tp"]
                totals[k]["fp"] += len(v["fp"])
                totals[k]["fn"] += len(v["fn"])
            bad = {k: {"fp": v["fp"], "fn": v["fn"]} for k, v in sc.items() if v["fp"] or v["fn"]}
            if bad:
                errs.append(f"filters {json.dumps(bad, ensure_ascii=False)}")
        prequal_ms.append(res["prequal_ms"])
        for k, v in res["call_ms"].items():
            call_ms[k].append(v)
        if res["pred"]["route"] == "products_only" and "extractor_ms" in res:
            path_ms.append(res["prequal_ms"] + res["extractor_ms"] + res["retrieval_ms"])
        for k, s in res["status"].items():
            statuses[f"{k}:{s}"] += 1
        for k, n in res["prompt_tokens"].items():
            tokens[k].append(n)
        mark = "ok " if not errs else "ERR"
        print(f"{mark} {row['id']} {res['prequal_ms']:6.0f}ms  {row['query'][:45]!s:45s} -> {res['query_en'][:70]}")
        for e in errs:
            print(f"      {e}")

    n = len(rows)
    print(f"\n=== prequal eval, {n} conversations ({args.mode} mode, extractor={args.extractor}) ===")
    for k in LABELS:
        acc = label_hits[k] / n
        target = TARGETS.get(k)
        verdict = f"  target >= {target:.2f} {'PASS' if acc >= target else 'FAIL'}" if target else ""
        print(f"{k:16s} accuracy {acc:6.3f} ({label_hits[k]}/{n}){verdict}")
    print(f"{'rewrite check':16s} pass     {rewrite_pass / n:6.3f} ({rewrite_pass}/{n})")
    print("  by tag: " + ", ".join(f"{t} {tag_pass[t]}/{tag_total[t]}" for t in sorted(tag_total)))

    if n_e2e:
        print(f"\nend-to-end rewrite check: extractor ({args.extractor}) on query_en, {n_e2e} rows")
        print(f"{'key':16s} {'prec':>6s} {'recall':>6s} {'f1':>6s} {'tp':>4s} {'fp':>4s} {'fn':>4s}")
        agg = Counter()
        for k in ALL_KEYS:
            c = totals[k]
            if not (c["tp"] or c["fp"] or c["fn"]):
                continue
            p, r, f1 = prf(c)
            print(f"{k:16s} {p:6.3f} {r:6.3f} {f1:6.3f} {c['tp']:4d} {c['fp']:4d} {c['fn']:4d}")
            agg.update(c)
        p, r, f1 = prf(agg)
        print(f"{'ALL (micro)':16s} {p:6.3f} {r:6.3f} {f1:6.3f} {agg['tp']:4d} {agg['fp']:4d} {agg['fn']:4d}"
              f"   target f1 >= 0.85 {'PASS' if f1 >= 0.85 else 'FAIL'}")

    if args.mode == "llm" or any(prequal_ms):
        p50, p95 = pct(prequal_ms, 50), pct(prequal_ms, 95)
        print(f"\nprequal latency: p50 {p50:.0f} ms, p95 {p95:.0f} ms, max {max(prequal_ms):.0f} ms"
              f"   targets: p50<=700 {'PASS' if p50 <= 700 else 'FAIL'}, p95<=1200 {'PASS' if p95 <= 1200 else 'FAIL'}")
        for k, v in call_ms.items():
            print(f"  {k:9s} p50 {pct(v, 50):5.0f} ms, p95 {pct(v, 95):5.0f} ms")
        if path_ms:
            p95p = pct(path_ms, 95)
            note = "" if args.extractor == "llm" else "  (extractor ran on rules, not the LLM: not a real path time)"
            print(f"products_only path (prequal+extractor||semantic+retrieval): p50 {pct(path_ms, 50):.0f} ms, p95 {p95p:.0f} ms"
                  f"   target p95<=3000 {'PASS' if p95p <= 3000 else 'FAIL'}{note}")
    print(f"call statuses: {dict(statuses)}")
    if tokens:
        print("prompt tokens (median, provider count): " +
              ", ".join(f"{k} {int(statistics.median(v))}" for k, v in tokens.items()))


def prf(c: Counter) -> tuple:
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 1.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


async def run(args) -> int:
    eval_path = Path(args.eval_file) if args.eval_file else EVAL_PATH
    rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in wanted]
    if args.limit:
        rows = rows[: args.limit]
    results_path = RESULTS_DIR / f"prequal_eval_results{'_' + eval_path.stem if args.eval_file else ''}.jsonl"

    if args.mode == "rescore":
        saved = {r["id"]: r for r in map(json.loads, results_path.read_text(encoding="utf-8").splitlines()) if r}
        rows = [r for r in rows if r["id"] in saved]
        results = saved
        for res in results.values():    # re-apply the current code-side router checks
            if res["status"].get("router") in ("ok", "fallback_model"):
                routed, _ = validate_route({k: res["pred"].get(k) for k in ROUTER_KEYS})
                res["pred"].update(routed)
        if args.extractor == "rules":   # re-run the offline extractor on the saved rewrites
            for row in rows:
                res = results[row["id"]]
                if row.get("filters") is not None:
                    f = MetadataFilters() if res["skip_metadata_filters"] else rules_only(res["query_en"], [])
                    res["filters"] = predicted(f)
    else:
        get_cache().clear()
        print(f"warm-up: {await warm_up()} pooled connections opened")
        pacer = TokenPacer(args.tpm)
        results = {}
        for i, row in enumerate(rows, 1):
            results[row["id"]] = await run_row(row, args, pacer)
            print(f"[{i:2d}/{len(rows)}] {row['id']} {results[row['id']]['prequal_ms']:.0f}ms", flush=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_path.write_text("".join(json.dumps(results[r["id"]], ensure_ascii=False) + "\n" for r in rows),
                                encoding="utf-8")
    print()
    score(rows, results, args)
    print(f"\nper-row results: {results_path.relative_to(ROOT)}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate the pre-qualification stage")
    ap.add_argument("--mode", choices=["llm", "rescore"], default="llm")
    ap.add_argument("--extractor", choices=["rules", "llm", "off"], default="rules",
                    help="how to run the extractor for the end-to-end check (default: offline rules)")
    ap.add_argument("--eval-file", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--tpm", type=int, default=6500, help="prompt-token-per-minute pacing budget (0 = no pacing)")
    return asyncio.run(run(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
