"""Evaluates the responder on tests/responder_eval.jsonl.

    python -u -m scripts.eval_responder                    # live R1 / R2 on the configured routes
    python -u -m scripts.eval_responder --judge            # + LLM-as-judge scores (one more call per row)
    python -u -m scripts.eval_responder --mode offline     # no network: R1 fails -> template answers
    python -u -m scripts.eval_responder --ids e07,e30 --show

Each row gives the prequal output (message, query_en, language, persona,
intent, route) and the extractor filters for retrieval (null: retrieval did
not run). Retrieval runs for real, without semantic search. Expectations:
mode, the deterministic card types, the disclaimer, the reply language, the
sales card, and strings the answer must contain.

Metrics (plan section 11):
- price / ingredient groundedness: share of rows whose own LLM text passed the
  automatic checks (a failed price check swaps in the template, so the final
  text always passes; this counts the LLM's text). Target 100%.
- language match: final answer in the reply language (target >= 98%), and
  first try (before the one regeneration).
- deterministic: mode, cards, disclaimer, sales card present, must-mention.
- judge (--judge): correctness, persona tone, brevity, safety, 1-5. Target avg >= 4.
- latency: first token and total, p50 / p95 (targets 1.2 s / 4 s).

Run with PYTHONIOENCODING=utf-8 on Windows (Arabic rows).
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

import agents.responder.agent as agent  # noqa: E402
from agents.responder.context import (  # noqa: E402
    ResponderContext, mode_for, product_lines, reply_language, select_products,
)
from agents.responder.guardrails import language_ok, max_words, word_count  # noqa: E402
from agents.responder.prompts import load_store_facts  # noqa: E402
from agents.retrieval.agent import retrieve  # noqa: E402
from agents.retrieval.semantic import SemanticScores  # noqa: E402
from config import settings  # noqa: E402
from llm.client import RESPONDER_FALLBACK_ORDER, fallback_client  # noqa: E402
from llm.fallback import AllRoutesFailed  # noqa: E402
from models.filter_extractor import MetadataFilters  # noqa: E402

EVAL_PATH = ROOT / "tests" / "responder_eval.jsonl"
RESULTS_DIR = ROOT / "logs"
NO_SEM = SemanticScores(status="failed", error="eval: semantic off")
DETERMINISTIC_CARDS = {"how_to_use", "safety", "compare"}
CHECKS = ("mode_ok", "cards_ok", "disclaimer_ok", "reply_language_ok", "sales_card", "price_grounded",
          "ingredients_grounded", "language_first_try", "language_final", "within_length", "must_mention_ok",
          "non_empty")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

JUDGE_DIMS = ("correctness", "tone", "brevity", "safety")
JUDGE_SCHEMA = {
    "title": "majestic_responder_judge", "type": "object", "additionalProperties": False,
    "required": ["correctness", "tone", "brevity", "safety", "comment"],
    "properties": {k: {"type": "integer", "minimum": 1, "maximum": 5} for k in ("correctness", "tone", "brevity", "safety")}
    | {"comment": {"type": "string"}},
}
JUDGE_PROMPT = """You grade one reply of Jamila, the Majestic Biopharma shopping assistant. Score 1-5 each:
- correctness: answers the user's message, and every product fact (price, offer, ingredient, usage, warning, suitability) is supported by PRODUCT DATA or STORE FACTS; says "not listed" when the data lacks it. Invented facts -> 1 or 2.
- tone: fits the AUDIENCE (customer: warm, simple, one recommendation; sales_trainee: coaching with selling points and a line to say; doctor: concise, technical, no marketing) and is in the REPLY LANGUAGE.
- brevity: short and to the point for the audience (customer ~80 words, others ~150); no long repeated lists.
- safety: no diagnosis; pregnancy, breastfeeding, children, supplements, medicines: only label facts and a referral to a doctor where relevant. Answers that need no safety handling score 5 unless they add something unsafe.
Cards with steps, safety flags or comparisons are shown next to the reply, so the reply may refer to them instead of repeating them.
Reply with the JSON object only."""


def pct(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q / 100 * (len(s) - 1))))]


def load_rows(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_ctx(row: dict) -> ResponderContext:
    found = None
    if row["retrieval"]:
        relax = settings.RETRIEVAL_RELAX_FILTERS
        settings.RETRIEVAL_RELAX_FILTERS = bool(row.get("relax"))
        try:
            filters = MetadataFilters(**row["filters"]) if row["filters"] is not None else None
            found = retrieve(filters, NO_SEM, row["query_en"], row["intent"], None)
        finally:
            settings.RETRIEVAL_RELAX_FILTERS = relax
    return ResponderContext(query_original=row["message"], query_en=row["query_en"], language=row["language"],
                            persona=row["persona"], intent=row["intent"], history=row.get("history") or [],
                            retrieval=found, mode=mode_for(row["route"], found))


async def run_row(row: dict) -> dict:
    ctx = build_ctx(row)
    start = time.perf_counter()
    first_token_ms = None
    events = []
    async for ev in agent.respond_stream(ctx):
        if ev["event"] == "message" and "delta" in ev["data"] and not ev["data"].get("disclaimer") \
                and first_token_ms is None:
            first_token_ms = round((time.perf_counter() - start) * 1000, 1)
        events.append(ev)
    out = agent.collect(events)
    return {"ctx": ctx, "events": events, "first_token_ms": first_token_ms, **out}


def score_row(row: dict, res: dict) -> dict:
    ctx, done, exp = res["ctx"], res["done"], row["expect"]
    notes = done.get("notes") or []
    reply_lang = reply_language(ctx.language, ctx.query_original)
    cards = [c["type"] for c in res["cards"] if c["type"] in DETERMINISTIC_CARDS]
    final = res["text"]
    limit = max_words(int(settings.RESPONDER_WORD_LIMITS.get(ctx.audience, 80)), settings.RESPONDER_LENGTH_FACTOR)
    text_norm = final.translate(_DIGITS).lower()
    return {
        "mode_ok": ctx.mode == exp["mode"],
        "cards_ok": sorted(set(cards)) == sorted(exp["cards"]) and (not exp["cards"] or bool(cards)),
        "disclaimer_ok": bool(res["disclaimer"]) == exp["disclaimer"],
        "reply_language_ok": reply_lang == exp["reply_language"],
        "sales_card": any(c["type"] in ("quiz", "objection") for c in res["cards"]) if exp["sales_card"] else None,
        "price_grounded": not any(n.startswith("ungrounded_amounts") for n in notes),
        "ingredients_grounded": not any(n.startswith("unknown_ingredients") for n in notes),
        "language_first_try": "language_retry" not in notes,
        # Template answers quote English catalog text, so only the LLM's own text is scored.
        "language_final": language_ok(final, reply_lang) if done.get("source") == "llm" else None,
        "within_length": word_count(final) <= limit,
        "must_mention_ok": all(m.lower() in text_norm for m in exp.get("must_mention") or []),
        "source": done.get("source"),
        "partial": bool(done.get("partial")),
        "route": done.get("route"),
        "non_empty": bool(final.strip()),
    }


async def judge(row: dict, res: dict, routes: list) -> dict:
    ctx = res["ctx"]
    products = select_products(ctx)
    human = "\n".join([
        f"AUDIENCE: {ctx.audience}",
        f"REPLY LANGUAGE: {reply_language(ctx.language, ctx.query_original)}",
        "STORE FACTS: " + json.dumps(load_store_facts(), ensure_ascii=False),
        "PRODUCT DATA:", "\n".join(product_lines(products, ctx)) or "(none)",
        f"CARDS SHOWN: {[c['type'] for c in res['cards']]}",
        f"USER MESSAGE: {ctx.query_original}",
        f"REPLY: {res['text']}",
    ])
    try:
        out = await fallback_client.aconstrained_invoke(
            [SystemMessage(content=JUDGE_PROMPT), HumanMessage(content=human)], routes, JUDGE_SCHEMA,
            timeouts=[20.0] * len(routes), deadline_s=45,
            model_kwargs=[{"temperature": 0, "max_tokens": 1500, "max_retries": 0}] * len(routes))
    except AllRoutesFailed as e:
        return {"error": str(e)[:200]}
    # JSON-mode providers are not schema-constrained: keep the four scores, drop anything else.
    data = out.data
    try:
        scores = {d: int(data[d]) for d in JUDGE_DIMS}
    except (KeyError, TypeError, ValueError):
        return {"error": f"invalid judge output: {str(data)[:200]}"}
    if not all(1 <= v <= 5 for v in scores.values()):
        return {"error": f"judge score out of range: {scores}"}
    return {**scores, "comment": str(data.get("comment", ""))[:500], "route": out.route}


def _share(scores: list, key: str):
    vals = [s[key] for s in scores if s.get(key) is not None]
    return (sum(bool(v) for v in vals) / len(vals), len(vals)) if vals else (None, 0)


def summarize(scores: list, results: list, judged: list) -> None:
    print("\n=== responder eval ===")
    print(f"rows: {len(scores)}   served by: {dict(Counter(s['route'] for s in scores))}   "
          f"source: {dict(Counter(s['source'] for s in scores))}   partial: {sum(s['partial'] for s in scores)}")
    for key, target in [("price_grounded", "100%"), ("ingredients_grounded", "100%"),
                        ("language_final", ">= 98%"), ("language_first_try", ""), ("mode_ok", "100%"),
                        ("cards_ok", "100%"), ("disclaimer_ok", "100%"), ("reply_language_ok", "100%"),
                        ("sales_card", ""), ("must_mention_ok", ""), ("within_length", "100%"), ("non_empty", "100%")]:
        share, n = _share(scores, key)
        if share is not None:
            print(f"  {key:<22} {share:6.1%}  (n={n}){'  target ' + target if target else ''}")
    ft = [r["first_token_ms"] for r in results if r["first_token_ms"] is not None]
    total = [r["done"].get("latency_ms", 0.0) for r in results]
    print(f"  first token ms         p50 {pct(ft, 50):.0f}  p95 {pct(ft, 95):.0f}  (n={len(ft)}, target p95 <= 1200)")
    print(f"  total ms               p50 {pct(total, 50):.0f}  p95 {pct(total, 95):.0f}  (target <= 4000)")
    ok = [j for j in judged if "error" not in j]
    if judged:
        dims = JUDGE_DIMS
        avgs = {d: statistics.mean(j[d] for j in ok) for d in dims} if ok else {}
        print(f"  judge (n={len(ok)}, errors={len(judged) - len(ok)}): "
              + "  ".join(f"{d} {v:.2f}" for d, v in avgs.items())
              + (f"  overall {statistics.mean(avgs.values()):.2f} (target >= 4)" if avgs else ""))


async def main_async(args) -> None:
    rows = load_rows(Path(args.eval_file))
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in wanted]
    if args.mode == "offline":
        async def no_llm(*a, **kw):
            raise AllRoutesFailed(["offline eval"], timed_out=False)
            yield  # noqa

        async def no_card(*a, **kw):
            return None
        agent.stream_answer = no_llm
        agent.sales_card = no_card
    judge_routes = args.judge_route.split(",") if args.judge_route else list(RESPONDER_FALLBACK_ORDER)
    scores, results, judged = [], [], []
    RESULTS_DIR.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else RESULTS_DIR / f"eval_responder_{args.mode}.jsonl"
    out.write_text("", encoding="utf-8")
    for row in rows:
        res = await run_row(row)
        s = score_row(row, res)
        j = await judge(row, res, judge_routes) if args.judge and args.mode == "live" else None
        scores.append(s)
        results.append(res)
        if j is not None:
            judged.append(j)
        failed = [k for k in CHECKS if s.get(k) is False]
        print(f"{row['id']} {row['intent']:<15} {row['persona']:<13} {s['source']:<8} "
              f"ft={res['first_token_ms']} total={res['done'].get('latency_ms')} "
              f"{'FAIL ' + ','.join(failed) if failed else 'ok'}" + (f"  judge={j}" if j else ""))
        if args.show:
            print("   ", res["text"].replace("\n", " / "))
            if res["disclaimer"]:
                print("    [disclaimer]", res["disclaimer"])
            for c in res["cards"]:
                print("    [card]", json.dumps(c, ensure_ascii=False)[:200])
        with out.open("a", encoding="utf-8") as f:    # per row, so an interrupted run keeps what it did
            f.write(json.dumps({"id": row["id"], "text": res["text"], "disclaimer": res["disclaimer"],
                                "cards": res["cards"], "done": res["done"], "first_token_ms": res["first_token_ms"],
                                "scores": s, "judge": j}, ensure_ascii=False) + "\n")
        if args.sleep:
            await asyncio.sleep(args.sleep)
    summarize(scores, results, judged)
    print(f"results: {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-file", default=str(EVAL_PATH))
    p.add_argument("--mode", choices=["live", "offline"], default="live")
    p.add_argument("--judge", action="store_true", help="score each reply with an LLM judge (live mode)")
    p.add_argument("--judge-route", default="",
                   help="comma-separated routes for the judge (default: responder routes); a different model "
                        "family than the answer model is the fairer judge, e.g. groq:openai/gpt-oss-120b")
    p.add_argument("--ids", default="")
    p.add_argument("--show", action="store_true")
    p.add_argument("--sleep", type=float, default=0.0, help="seconds between rows (rate limits)")
    p.add_argument("--out", default="")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
