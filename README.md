# Majestic metadata filter extractor

Turns one English user message into metadata filters for the Majestic
Biopharma catalog. It runs 10 small LLM calls in parallel, each on one key
with its own schema, merges the results in code, and returns one
`MetadataFilters` object for retrieval. It is built on the same `llm/` layer,
config style and prompt layout as the Egyptian drug assistant; see
[ARCHITECTURE_NOTES.md](ARCHITECTURE_NOTES.md).

```python
import asyncio
from agents.filter_extractor.agent import extract_filters, warm_up

async def main():
    await warm_up()   # optional: pre-open pooled connections
    f = await extract_filters("pregnant patient, oily acne-prone skin, no salicylic acid",
                              context=["Vacation Sebio-Control Acne Cream 60ml"])
    print(f.model_dump_json(indent=2))

asyncio.run(main())
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # set GROQ_API (and optionally the EXTRACTOR_* settings)
make test                   # offline unit tests
make eval-rules             # offline eval of the rule-based fallback
make eval                   # live eval: per-key precision/recall, exact match, p50/p95
```

On Windows without `make`, run the underlying commands directly, for example
`python -m pytest` or `python -u -m scripts.eval_extractor --mode llm`.

## Configuration

All settings are environment variables read in `config.py`.

| Variable | Default | Meaning |
|---|---|---|
| `EXTRACTOR_PRIMARY_ROUTE` | `groq:qwen/qwen3.8-27b` | `router:model` for the first attempt |
| `EXTRACTOR_FALLBACK_ROUTE` | `groq:openai/gpt-oss-20b` | tried once if the primary fails |
| `EXTRACTOR_USE_FALLBACK_MODEL` | `true` | turns the fallback model on or off |
| `EXTRACTOR_USE_RULE_FALLBACK` | `true` | turns the deterministic per-key fallback on or off |
| `EXTRACTOR_TIMEOUT_S` / `EXTRACTOR_FALLBACK_TIMEOUT_S` | `2.5` / `2.5` | per-attempt timeouts |
| `EXTRACTOR_CALL_DEADLINE_S` | `4.5` | hard cap per call, across both attempts |
| `EXTRACTOR_CALL_TIMEOUTS` | `{}` | per-call primary timeout overrides, as JSON |
| `EXTRACTOR_MAX_TOKENS` | the plan's table | per-call `max_tokens` overrides, as JSON |
| `EXTRACTOR_TRANSLATE` | `false` | adds a rewrite-to-English call before the fan-out |
| `EXTRACTOR_ALLOWED_IN_PROMPT` | `true` | set `false` only if the provider enforces schema enums well |
| `EXTRACTOR_CACHE_TTL_S` / `EXTRACTOR_CACHE_SIZE` | `86400` / `2048` | result cache |

## Regenerating after a new scrape

No allowed value is hard-coded. The prompts' allowed lists, the schema enums,
the hierarchy, the product-line and hero maps, and the name indexes are all
built from `data/majestic_catalog.json` and `data/metadata_catalog.json` at
startup.

1. **Replace the raw scrape.** Put the new scrape at `data/data.json`.
2. **Rebuild the two catalog files.**

   ```bash
   python -m scripts.build_catalogs --check   # lists new ingredient spellings, brands, missing product lines
   python -m scripts.build_catalogs           # writes the two catalog files
   ```

   If the check lists new raw ingredient spellings that should merge into an
   existing name, add them to `ingredient_canon` in `data/catalog_rules.json`.
   A new brand also needs an entry in `brand_aliases` there. Rerun the build
   afterwards.
3. **Restart the service.** Nothing else changes.

If you already have the two generated JSON files from elsewhere, drop them
into `data/` and restart.

`data/aliases.json` holds English synonyms and typos for the rule-based
fallback only. Values that no longer exist in the catalog are ignored
automatically.

## Evaluation set

`tests/eval_queries.jsonl` has 68 labeled English queries. 28 of them contain
typos. They cover customers, sales trainees and doctors, 6 greetings or
off-topic messages, and 2 follow-ups that use context. Each row has
`expected` values and `optional` ones. Optional values are neither rewarded
nor penalized; they cover judgment calls such as a parent category implied by
a product type, or a concern implied by a skin type. Names are scored through
`matched_handles`.

## Latest results

Holdout run on 2026-09-26: `python -u -m scripts.eval_extractor --mode llm --eval-file tests/eval_holdout.jsonl`,
20 queries. Per-query output is in `logs/eval_results_llm_eval_holdout.jsonl`.

| Metric | Result | Target | Status |
|---|---|---|---|
| Catalog precision (micro) | 0.881 | ≥ 0.90 | FAIL |
| Catalog recall (micro) | 0.860 | ≥ 0.85 | PASS |
| Exact-match rate | 13/20 = 0.650 | – | – |
| Latency p50 | 671 ms | ≤ 1000 ms | PASS |
| Latency p95 | 1819 ms | ≤ 2000 ms | PASS |
| Latency max | 2909 ms | – | – |

Latency counts only queries that did not short-circuit.

### Per key

| Key | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|
| brand | 1.000 | 1.000 | 5 | 0 | 0 |
| category | 1.000 | 1.000 | 0 | 0 | 0 |
| product_group | 0.000 | 1.000 | 0 | 1 | 0 |
| product_type | 1.000 | 0.889 | 8 | 0 | 1 |
| product_form | 0.889 | 1.000 | 8 | 1 | 0 |
| concerns | 1.000 | 0.667 | 6 | 0 | 3 |
| suitable_for | 0.700 | 1.000 | 7 | 3 | 0 |
| hero_ingredient | 1.000 | 0.000 | 0 | 0 | 2 |
| include | 1.000 | 1.000 | 3 | 0 | 0 |
| exclude | 1.000 | 1.000 | 0 | 0 | 0 |
| handles | 1.000 | 0.800 | 4 | 0 | 1 |
| **Catalog (micro)** | **0.881** | **0.860** | **37** | **5** | **6** |

A key with no expected and no predicted values scores 1.000.

### Per call

| Call | p50 ms | p95 ms | Prompt tokens | Budget | Over budget |
|---|---|---|---|---|---|
| names | 510 | 1021 | 434 | 261 | +66% |
| brand | 432 | 1813 | 398 | 207 | +92% |
| category | 460 | 1420 | 333 | 187 | +78% |
| product_group | 536 | 661 | 450 | 231 | +95% |
| product_type | 473 | 984 | 631 | 342 | +85% |
| product_form | 520 | 1309 | 438 | 216 | +103% |
| concerns | 547 | 711 | 629 | 334 | +88% |
| suitable_for | 485 | 790 | 479 | 267 | +79% |
| hero_ingredient | 419 | 676 | 539 | 284 | +90% |
| ingredients | 358 | 421 | 4011 | 1185 | +238% |

Call statuses across all 200 calls: `fallback_model` 127, `rule_based` 53,
`ok` 10, `skipped` 10.

### Notes

- Most calls (127 of 200) were answered by the fallback model and 53 by the
  rule-based fallback. Only 10 were answered by the primary model, so these
  numbers mostly describe the fallbacks, not the primary route.
- Precision misses the target because of `suitable_for` (3 false positives),
  `product_group` (1) and `product_form` (1).
- Recall is weakest on `hero_ingredient` (0 of 2 found) and `concerns`
  (6 of 9 found).
- Every call's prompt is over its token budget. `ingredients` is the largest,
  at 4011 tokens against a budget of 1185.
