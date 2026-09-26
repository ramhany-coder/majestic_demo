"""End-to-end smoke test: raw Arabic / Arabizi / English messages through the
whole chat pipeline (live LLM calls):

    prequal (rewriter || router) -> extractor || semantic search -> retrieval -> list or responder

    python -m scripts.smoke_e2e              # paced for the Groq free tier (about 7.5k tokens per turn)
    python -m scripts.smoke_e2e --tpm 0      # no pacing (paid tier)

Every message runs in a fresh session, so no cache helps it. For each turn it
checks the path, k, the expected products and the forbidden ones, and reports
the products_only path latency (target: p95 <= 3 s): prequal + extractor ||
semantic + retrieval, over every turn the router sent to products_only
(a relaxed result then goes to the responder, whose time is left out). The
pacing waits happen between turns and are not counted.
"""

import argparse
import asyncio
import statistics
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.filter_extractor.agent import get_cache as extractor_cache  # noqa: E402
from agents.orchestrator.orchestrator import handle_message, warm_up  # noqa: E402
from agents.prequal.session_context import SessionStore  # noqa: E402
from scripts.eval_extractor import TokenPacer  # noqa: E402

TURN_TOKENS = 7500
TARGET_P95_MS = 3000

# message, expected k, handles one of which must be in the top 3, handles that must never appear
CASES = [
    ("عايزة سيروم للشعر من غير سيليكون", 10,
     {"capixy-hair-serum", "capixy-anti-dandruff-serum-spray-120ml"}, {"methytral-esthetic-cream-50gm", "methytral-hyalu-gel-50gm"}),
    ("warreeni 3 sun blok spray lel bashra el dohneya", 3,
     {"vacation-sunscreen-pure-touch-hydro-gel-60-ml", "vacation-unseen-sunscreen-gel-30ml", "vacation-sunscreen-lotion-120ml",
      "vacation-sunscreen-lotion-spray-200ml"}, set()),
    ("كابكسي فوم بيتستخدم ازاي؟", 10, {"capixy-dry-foam-120-ml"}, set()),
    ("انا حامل وعايزة سيروم للوش من غير ريتينول ولا ساليسيليك", 10,
     {"vacation-hyaluronic-acid-serum-3-30-ml", "vacation-vitamin-c-10-30-ml", "vacation-argireline-serum-10-30ml",
      "vacation-copper-peptide-serum-10-30ml", "vacation-sebio-control-glycolic-acid-30ml"},
     {"vacation-retinol-serum-1-30-ml", "vacation-eye-cream", "vacation-wrinkless-anti-aging-jelly-cream-60-gm",
      "vacation-niacinamide-10-zinc-pca-serum-30-ml"}),
    ("3ayza deodorant spray lel 3ara2", 10,
     {"vacation-deodorant-spray-icy-sage-200-ml", "vacation-deodorant-spray-lyccee-pink-pepper-200-ml",
      "vacation-deodorant-spray-raspberry-marshmallow-200-ml", "vacation-deodorant-spray-sweet-saffron-200-ml",
      "vacation-watermelon-deodrant-sparay-200-ml", "vacation-antiperspirant-fragrance-free-200-ml"}, set()),
    ("what's the difference between capixi dray foom and the capixy vials?", 10,
     {"capixy-dry-foam-120-ml", "capixy-anti-hair-loss-vials-70ml"}, set()),
]


async def run(args) -> int:
    print(f"warm-up: {await warm_up()} pooled connections opened (embedding model loaded)")
    extractor_cache().clear()
    pacer = TokenPacer(args.tpm)
    store = SessionStore()
    failures, path_ms = 0, []
    for message, want_k, want_top, forbidden in CASES:
        await pacer.wait(TURN_TOKENS)
        turn = await handle_message(message, f"smoke-{uuid.uuid4().hex[:8]}", store)
        used = sum(turn.prequal.meta.get("prompt_tokens", {}).values())
        used += sum((turn.filters or {}).get("meta", {}).get("prompt_tokens", {}).values())
        pacer.record(used or TURN_TOKENS)

        handles = [p["handle"] for p in turn.products]
        r = turn.retrieval or {}
        problems = []
        if turn.prequal.k != want_k:
            problems.append(f"k={turn.prequal.k} (want {want_k})")
        if not set(handles[:3]) & want_top:
            problems.append(f"none of {sorted(want_top)[:3]}... in the top 3")
        if set(handles) & forbidden:
            problems.append(f"forbidden returned: {sorted(set(handles) & forbidden)}")
        if turn.prequal.route == "products_only":
            path_ms.append(turn.timings_ms["total"] - turn.timings_ms.get("responder", 0.0))
        failures += bool(problems)
        t = turn.timings_ms
        print(f"\n{'ok ' if not problems else 'ERR'} {message}")
        print(f"    query_en={turn.prequal.query_en!r} intent={turn.prequal.intent} route={turn.prequal.route} "
              f"k={turn.prequal.k} path={turn.path}")
        print(f"    timings: total {t['total']:.0f} ms = prequal {t['prequal']:.0f} + extractor||semantic "
              f"{t.get('extractor_semantic', 0):.0f} (extractor {t.get('extractor', 0):.0f}, semantic {t.get('semantic', 0):.0f}) "
              f"+ retrieval {t.get('retrieval', 0):.1f}" + (f" + responder {t['responder']:.0f}" if "responder" in t else ""))
        print(f"    relaxed={r.get('relaxed_keys')} names={r.get('name_hits', [])[:3]} fallback={r.get('fallback')} "
              f"semantic={r.get('meta', {}).get('semantic')}")
        print(f"    top: {handles[:5]}")
        for p in problems:
            print(f"    !! {p}")

    print(f"\n=== smoke test: {len(CASES) - failures}/{len(CASES)} turns as expected ===")
    if path_ms:
        s = sorted(path_ms)
        p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
        print(f"products_only path (prequal + extractor||semantic + retrieval) over {len(s)} turns: p50 {statistics.median(s):.0f} ms, p95 {p95:.0f} ms, max {s[-1]:.0f} ms"
              f"   target p95 <= {TARGET_P95_MS} ms {'PASS' if p95 <= TARGET_P95_MS else 'FAIL'}")
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="End-to-end smoke test (live LLM calls)")
    ap.add_argument("--tpm", type=int, default=6500, help="prompt tokens per minute to pace to (0 = no pacing)")
    return asyncio.run(run(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
