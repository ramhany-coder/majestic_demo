"""Evaluates the metadata filter extractor on tests/eval_queries.jsonl.

    python -m scripts.eval_extractor                 # live LLM run (paced for the rate limit)
    python -m scripts.eval_extractor --mode rules    # offline: rule-based fallback only
    python -m scripts.eval_extractor --mode rescore  # re-merge saved LLM call outputs with current code
    python -m scripts.eval_extractor --limit 10 --ids q01,q13

Scoring: per key, a predicted value is a true positive if it is in `expected`,
ignored if it is in `optional`, else a false positive; an expected value not
predicted is a false negative. Optional values cover things a reasonable
annotator could go either way on (implied parent categories, a concern implied
by a skin type). A query is an exact match when it has no FP and no FN on any
key. Names are scored through `matched_handles` (key "handles").

Targets (plan section 10): precision >= 0.90 and recall >= 0.85 over the
catalog keys, p95 latency <= 2.0 s.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.filter_extractor.agent import extract_filters, get_cache, warm_up  # noqa: E402
from agents.filter_extractor.merge import merge_outputs  # noqa: E402
from agents.filter_extractor.rule_based import extract_rule_based  # noqa: E402
from agents.filter_extractor.schemas import CALL_SPECS, validate_output  # noqa: E402
from agents.filter_extractor.agent import is_small_talk  # noqa: E402
from models.filter_extractor import MetadataFilters  # noqa: E402

EVAL_PATH = ROOT / "tests" / "eval_queries.jsonl"
RESULTS_DIR = ROOT / "logs"
CATALOG_KEYS = ["brand", "category", "product_group", "product_type", "product_form", "concerns",
                "suitable_for", "hero_ingredient", "include", "exclude"]
ALL_KEYS = CATALOG_KEYS + ["handles"]
PROMPT_BUDGETS = {"names": 261, "brand": 207, "category": 187, "product_group": 231, "product_type": 342,
                  "product_form": 216, "concerns": 334, "suitable_for": 267, "hero_ingredient": 284, "ingredients": 1185}


def predicted(f: MetadataFilters) -> dict:
    return {
        "brand": f.brand, "category": f.category, "product_group": f.product_group,
        "product_type": f.product_type, "product_form": f.product_form, "concerns": f.concerns,
        "suitable_for": f.suitable_for, "hero_ingredient": f.hero_ingredient,
        "include": f.ingredients.include, "exclude": f.ingredients.exclude, "handles": f.matched_handles,
    }


def score_row(row: dict, pred: dict) -> dict:
    per_key = {}
    for k in ALL_KEYS:
        exp = set(row["expected"].get(k, []))
        opt = set(row["optional"].get(k, [])) - exp
        got = set(pred.get(k, []))
        tp = got & exp
        fp = got - exp - opt
        fn = exp - got
        per_key[k] = {"tp": len(tp), "fp": sorted(fp), "fn": sorted(fn)}
    return per_key


def rules_only(query: str, context) -> MetadataFilters:
    t = time.perf_counter()
    if is_small_talk(query):
        f = MetadataFilters()
        f.meta.short_circuit = "small_talk"
    else:
        outs = {s.key: validate_output(s, extract_rule_based(s.key, query, context))[0] for s in CALL_SPECS}
        f = merge_outputs(outs, query=query, context=context)
        f.meta.call_outputs = outs
        f.meta.calls = {s.key: "rule_based" for s in CALL_SPECS}
    f.meta.latency_ms = round((time.perf_counter() - t) * 1000, 1)
    return f


def pct(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round(p / 100 * (len(values) - 1)))))
    return values[k]


class TokenPacer:
    """Keeps prompt tokens under a tokens-per-minute budget (Groq free tier: 7000 ITPM)."""

    def __init__(self, tpm: int):
        self.tpm = tpm
        self.window = deque()  # (timestamp, tokens)

    async def wait(self, next_tokens: int):
        if self.tpm <= 0:
            return
        while True:
            now = time.monotonic()
            while self.window and now - self.window[0][0] > 60:
                self.window.popleft()
            used = sum(t for _, t in self.window)
            if used + next_tokens <= self.tpm or not self.window:
                return
            await asyncio.sleep(max(0.5, 60 - (now - self.window[0][0]) + 0.5))

    def record(self, tokens: int):
        self.window.append((time.monotonic(), tokens))


async def run(args) -> int:
    eval_path = Path(args.eval_file) if args.eval_file else EVAL_PATH
    rows = [json.loads(l) for l in eval_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in wanted]
    if args.limit:
        rows = rows[: args.limit]

    if args.mode == "llm":
        get_cache().clear()
        opened = await warm_up()
        print(f"warm-up: {opened} pooled connections opened")
    saved = {}
    if args.mode == "rescore":
        src = Path(args.source) if args.source else RESULTS_DIR / "eval_results_llm.jsonl"
        saved = {r["id"]: r for r in map(json.loads, src.read_text(encoding="utf-8").splitlines()) if r}
    pacer = TokenPacer(args.tpm if args.mode == "llm" else 0)
    est_tokens = 4400

    totals = {k: Counter() for k in ALL_KEYS}
    latencies, exact, statuses = [], 0, Counter()
    call_lat = defaultdict(list)
    prompt_tokens = defaultdict(list)
    results = []
    for i, row in enumerate(rows, 1):
        if args.mode == "llm":
            await pacer.wait(est_tokens)
            f = await extract_filters(row["query"], row["context"], use_cache=False)
            used = sum(f.meta.prompt_tokens.values())
            if used:
                est_tokens = used
                pacer.record(used)
        elif args.mode == "rescore":
            old = saved[row["id"]]
            if old.get("call_outputs"):
                f = merge_outputs(old["call_outputs"], query=row["query"], context=row["context"])
            else:
                f = MetadataFilters()
                f.meta.short_circuit = "small_talk"
            f.meta.latency_ms = old["latency_ms"]
            f.meta.calls = old["calls"]
            f.meta.call_latency_ms = old.get("call_latency_ms", {})
            f.meta.prompt_tokens = old.get("prompt_tokens", {})
            f.meta.call_outputs = old.get("call_outputs", {})
        else:
            f = rules_only(row["query"], row["context"])
        pred = predicted(f)
        sc = score_row(row, pred)
        ok = all(not v["fp"] and not v["fn"] for v in sc.values())
        exact += ok
        for k, v in sc.items():
            totals[k]["tp"] += v["tp"]
            totals[k]["fp"] += len(v["fp"])
            totals[k]["fn"] += len(v["fn"])
        if not f.meta.short_circuit:
            latencies.append(f.meta.latency_ms)
        for key, st in f.meta.calls.items():
            statuses[st] += 1
        for key, ms in f.meta.call_latency_ms.items():
            call_lat[key].append(ms)
        for key, n in f.meta.prompt_tokens.items():
            prompt_tokens[key].append(n)
        errs = {k: {"fp": v["fp"], "fn": v["fn"]} for k, v in sc.items() if v["fp"] or v["fn"]}
        results.append({"id": row["id"], "query": row["query"], "exact": ok, "errors": errs,
                        "latency_ms": f.meta.latency_ms, "calls": f.meta.calls, "pred": pred,
                        "call_outputs": f.meta.call_outputs, "call_latency_ms": f.meta.call_latency_ms,
                        "prompt_tokens": f.meta.prompt_tokens, "notes": f.meta.notes})
        mark = "ok " if ok else "ERR"
        print(f"[{i:2d}/{len(rows)}] {mark} {f.meta.latency_ms:7.0f}ms {row['id']} {row['query'][:60]}"
              + ("" if ok else f"  {json.dumps(errs, ensure_ascii=False)}"))

    RESULTS_PATH = RESULTS_DIR / f"eval_results_{args.mode}{'_' + eval_path.stem if args.eval_file else ''}.jsonl"
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")

    def pr(c):
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 1.0
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 1.0
        return p, r

    print(f"\n=== {args.mode} mode, {len(rows)} queries ===")
    print(f"{'key':16s} {'prec':>6s} {'recall':>6s} {'tp':>4s} {'fp':>4s} {'fn':>4s}")
    agg = Counter()
    for k in ALL_KEYS:
        p, r = pr(totals[k])
        print(f"{k:16s} {p:6.3f} {r:6.3f} {totals[k]['tp']:4d} {totals[k]['fp']:4d} {totals[k]['fn']:4d}")
        if k in CATALOG_KEYS:
            agg.update(totals[k])
    p, r = pr(agg)
    print(f"{'CATALOG (micro)':16s} {p:6.3f} {r:6.3f} {agg['tp']:4d} {agg['fp']:4d} {agg['fn']:4d}"
          f"   targets: prec>=0.90 {'PASS' if p >= 0.9 else 'FAIL'}, recall>=0.85 {'PASS' if r >= 0.85 else 'FAIL'}")
    print(f"exact-match rate: {exact}/{len(rows)} = {exact / max(1, len(rows)):.3f}")
    if latencies:
        p50, p95 = pct(latencies, 50), pct(latencies, 95)
        print(f"latency (non-short-circuit): p50 {p50:.0f} ms, p95 {p95:.0f} ms, max {max(latencies):.0f} ms"
              + (f"   targets: p50<=1000 {'PASS' if p50 <= 1000 else 'FAIL'}, p95<=2000 {'PASS' if p95 <= 2000 else 'FAIL'}"
                 if args.mode in ("llm", "rescore") else ""))
    print(f"call statuses: {dict(statuses)}")
    if args.mode in ("llm", "rescore") and call_lat:
        print(f"\n{'call':16s} {'p50 ms':>7s} {'p95 ms':>7s} {'prompt tok':>10s} {'budget':>6s} {'over':>6s}")
        for s in CALL_SPECS:
            toks = prompt_tokens.get(s.key)
            med = int(statistics.median(toks)) if toks else 0
            b = PROMPT_BUDGETS[s.key]
            print(f"{s.key:16s} {pct(call_lat[s.key], 50):7.0f} {pct(call_lat[s.key], 95):7.0f} {med:10d} {b:6d} {med / b - 1:+6.0%}")
    print(f"\nper-query results: {RESULTS_PATH.relative_to(ROOT)}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate the metadata filter extractor")
    ap.add_argument("--mode", choices=["llm", "rules", "rescore"], default="llm")
    ap.add_argument("--source", default="", help="rescore: saved results file (default logs/eval_results_llm.jsonl)")
    ap.add_argument("--eval-file", default="", help="labeled queries (default tests/eval_queries.jsonl)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--tpm", type=int, default=6500, help="prompt-token-per-minute pacing budget (0 = no pacing)")
    return asyncio.run(run(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
