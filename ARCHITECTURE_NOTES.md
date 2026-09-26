# Architecture notes: Majestic metadata filter extractor

This module turns one English user message into metadata filters for the
Majestic Biopharma catalog (108 products). It runs as 10 small LLM calls in
parallel, merges their results in code, and hands one filter object to
retrieval.

## 1. What the existing project does (`C:\AI\Drug-assistant-agent`)

| Concern | How the drug assistant does it | Reused here as |
|---|---|---|
| LLM clients | LangChain chat models (`ChatAnthropic`, `ChatGroq`, `ChatOpenAI`, `ChatGoogleGenerativeAI`, `ChatOllama`), built by `llm/llm_models.py::Llm.get_model(router, model)` | Same file, extended with constructor overrides and a cached instance per config |
| Fallback chain | `llm/fallback.py::FallBack` walks a `fallback_order` list of routers. `constrained_invoke` does structured output and returns a dict. Only Groq is configured today (`FALLBACK_ORDER = ["groq"]`) | Same class, plus a new async `aconstrained_invoke` |
| Structured output | `llm.with_structured_output(PydanticModel, method="json_schema")`, which is native constrained decoding and avoids forced tool calls | Same method, with a JSON-schema dict built from the catalog so enums are never hard-coded |
| Config | `config.py::Settings` reads env vars through `python-dotenv`; the keys are `GROQ_API`, `ANTHROPIC_API_KEY` and so on | Same, plus the `EXTRACTOR_*` entries |
| Prompts | `prompts/<name>.md` with a `## The prompt` fenced block, loaded by `llm/prompt_loader.py`. Each agent has a `<agent>_prompt.py` that builds the messages | `prompts/filter_extractor/<call>.md`, same loader |
| Agents | `agents/<agent>/<agent>.py` exposes a node `fn(state) -> partial state dict`. LangGraph wires the nodes (`api/graph_builder.py`), and `api/workflow.py::_timed` records per-stage latency into `state["stage_timings"]` | `agents/filter_extractor/agent.py::filter_extractor(state)` (async) |
| Output models | Pydantic models in `models/` | `models/filter_extractor.py::MetadataFilters` |
| Logging | `logging` loggers (`"pipeline"`), plus `print` inside `FallBack` | `logging.getLogger("filter_extractor")` and `"llm.fallback"` |
| Tests | pytest, `tests/conftest.py` puts the repo root on `sys.path`, fixtures monkeypatch module globals, no network | Same style; the LLM is faked at `client_llm.get_cached_model` |

No new framework or SDK was added. The new dependency in `requirements.txt`
is `tiktoken`, used only for the offline prompt-budget test; it was already
installed in the drug-assistant environment.

### Changes to the copied `llm/` layer (backward compatible)

- **Routes may name a model.** `"groq:openai/gpt-oss-20b"` means router plus
  model, so one chain can try two models on the same provider. A plain `"groq"`
  still resolves through the constructor mapping, as before.
- **`Llm.get_model(router, model, **overrides)`** passes overrides such as
  `max_tokens`, `max_retries` and `reasoning_effort` to the LangChain
  constructor.
- **`Llm.get_cached_model(...)`** returns one instance per distinct config.
  Groq and OpenAI instances share one `httpx.AsyncClient` per event loop, so
  all 10 concurrent calls draw from a single connection pool.
- **`FallBack.aconstrained_invoke(...)`** is the async twin of
  `constrained_invoke`. It adds a timeout per route and a deadline for the
  whole chain, and it uses `include_raw=True` to read token usage. It returns
  a `ConstrainedResult` saying which route answered, and raises
  `AllRoutesFailed` (a `RuntimeError`) carrying the errors of every attempt.
- `invoke` and `constrained_invoke` are unchanged apart from accepting the new
  route syntax.

## 2. Module layout

```
agents/filter_extractor/
  catalog.py     load both JSON files -> allowed lists, hierarchy, name indexes, aliases
  schemas.py     the 10 CallSpecs, JSON schemas with enums, validate/snap (rapidfuzz >= 90)
  prompts.py     fill {{allowed}} / {{product_lines}}; split into static system + query-last human
  calls.py       10 async functions -> run_call(): chain -> validate -> rule-based fallback
  rule_based.py  deterministic extractor, one function per key
  merge.py       hierarchy check, hero -> ingredients, exclude wins, name -> handles
  grounding.py   drop values the query doesn't support (see section 4)
  cache.py       TTL LRU (24 h)
  agent.py       extract_filters() orchestrator, filter_extractor(state) node, warm_up()
prompts/filter_extractor/*.md   10 call prompts + optional translate prompt
models/filter_extractor.py      MetadataFilters output model
data/data.json                  raw scrape
data/catalog_rules.json         hand-kept canonicalisation rules
data/majestic_catalog.json      generated: products + ingredients_canonical
data/metadata_catalog.json      generated: allowed values, hierarchy, maps
data/aliases.json               English synonyms and typos for the rule-based fallback
scripts/build_catalogs.py       raw scrape -> the two catalog files
scripts/eval_extractor.py       eval runner (metrics table)
```

## 3. Runtime flow

1. **Input.** `extract_filters(query, context)` trims the query to 500
   characters. With `EXTRACTOR_TRANSLATE=true`, one extra call rewrites
   Arabic or Arabizi into English first. It is off by default because
   upstream (`translator_to_eng` in the drug-assistant graph) already produces
   `eng_query`.
2. **Short-circuit.** A message made only of greeting or thanks words, or only
   emoji, returns empty filters with `meta.short_circuit` set. No LLM call is
   made.
3. **Cache.** Results are cached by normalized query plus context for 24 h. A
   result built from any degraded call is not cached.
4. **Fan-out.** `asyncio.gather(*10 calls, return_exceptions=True)`. Only the
   names call receives the recent product names (`CONTEXT:`).
5. **Each call** runs through `FallBack.aconstrained_invoke` with this chain:
   - primary route, timeout `EXTRACTOR_TIMEOUT_S` (2.5 s)
   - fallback route, tried once, timeout `EXTRACTOR_FALLBACK_TIMEOUT_S`
   - the chain as a whole is capped by `EXTRACTOR_CALL_DEADLINE_S` (4.5 s)

   The response is validated against the call's schema. Off-list values are
   snapped with rapidfuzz (score 90 or more) or dropped, and lists are cut to
   `maxItems`. If both models fail, the rule-based extractor fills that key
   only. `run_call` never raises.
6. **Merge and post-process** in `merge.py`: hierarchy, grounding, hero
   expansion, exclude wins, name matching. `MetadataFilters` is returned. `meta.calls[key]` is one of these statuses:
   - `ok`: the primary model answered.
   - `fallback_model`: the fallback model answered.
   - `rule_based`: both LLM attempts errored, and the rules filled the key.
   - `timeout`: every LLM attempt timed out, and the rules filled the key.
   - `failed`: the rule fallback is disabled, so the key is empty.
   - `skipped`: the message was small talk.

### Models

| Role | Default route | Why |
|---|---|---|
| Primary | `groq:qwen/qwen3.8-27b` | Fastest structured-output model on this Groq account (about 0.15 to 0.3 s per call, no reasoning tokens) |
| Fallback | `groq:openai/gpt-oss-20b` | Smallest strict-schema model on Groq. It is a reasoning model, so it gets `reasoning_effort="low"` and `max_tokens + 300` |

Anthropic, OpenAI and Gemini keys are not configured in this project, which
matches the drug assistant's own comment. To switch, set
`EXTRACTOR_PRIMARY_ROUTE=anthropic:claude-haiku-4-5-20251001` or similar;
no code changes are needed.

## 4. Deviations from the plan, and why

- **Grounding checks** (`grounding.py`) were added to the merge step. The
  first live eval gave 0.867 catalog precision, below the 0.90 target. Almost
  every false positive came from the small model recommending instead of
  extracting:
  - inventing product names ("Capixy Anti Hair Loss Shampoo" for a query
    naming no brand)
  - adding types the user never asked for ("hair tonic" for "my hair is
    falling")
  - assuming body areas ("body", "scalp")
  - reading the brand "capixy" as the ingredient Capixyl

  The prompts already forbid all of this; the checks enforce it in code, and
  every drop is written to `meta.notes`. A value is kept only when the query
  (or the context, for names) supports it:
  - **Names:** the name's brand or product line must appear in the query or
    context, and a bare brand is not a name.
  - **Product type:** the type's head noun, or a known alias, must appear in
    the query.
  - **Area values** of suitable_for: an area word must appear in the query.
  - **Brands:** the brand must be mentioned in the query or context, or
    implied by a matched product.
- **Name matching** does not use `token_set_ratio >= 80`. That scorer rates
  any subset as 100, so "Capixy Tonic Spray" matched "Capixy Intense Tonic
  Spray", and a bare "Capixy" matched all 18 Capixy products. `NameMatcher`
  scores IDF-weighted fuzzy token overlap in both directions, and keeps only
  near-best candidates. A name that is a product line, such as
  "Sebio-Control", matches every product in that line.
- **Hierarchy runs before grounding.** Grounding may drop a recommended type,
  and a stated group must not be dropped along with it.
- **The product-line list** in the brand prompt is filled from the catalog
  (`{{product_lines}}`), not written into the prompt text.
- **Concentrations stay in the name index** ("Urea 15%" vs "Urea 5%").
  Only ml and gm sizes are stripped.

## 5. Hand-off contract for the retrieval layer

`MetadataFilters` maps onto product fields through
`metadata_catalog.json -> field_map`. Retrieval should apply it as follows.

- **Combining keys.** AND across keys, OR within a key.
- **Ingredients.** `ingredients.include` means all must be present in
  `ingredients_canonical`. `ingredients.exclude` means none may be present,
  and it is never relaxed.
- **Names.** `matched_handles` is the strongest signal. When it is non-empty,
  retrieval can restrict to those handles, or rank them first.
- **Relaxation.** On zero results, drop filters in this order until something
  matches: `product_form`, `suitable_for`, `concerns`, `product_group`,
  `category`. Never relax `exclude`.
- **Unmatched terms.** `unmatched.*` terms go to full-text search on
  `description` as soft signals. `unmatched.concerns` and `unmatched.include`
  boost. `unmatched.exclude` penalizes, but a match on "<term>-free" or
  "without <term>" in the description boosts instead.

## 6. Plugging into the drug-assistant graph

`filter_extractor` is an async node. Its input is `state["eng_query"]` (or
`state["query"]`), plus the recent product names from
`state["recent_product_names"]`, else the `name` fields of `state["context"]`.
Its output is `{"metadata_filters": {...}}`.

- **Wrapper.** LangGraph runs async nodes when the graph is invoked with
  `ainvoke`. The existing `_timed` wrapper is sync, so it needs an async twin
  that awaits `fn(state)`. Its body is otherwise identical.
- **State.** Add `metadata_filters: Optional[dict]` to `models/state.py`.
- **Warm-up.** Call `await warm_up()` in the FastAPI lifespan, next to the PII
  warm-up. It pre-opens pooled connections with a cheap model-list request,
  so the first user doesn't pay 10 TLS handshakes.

## 7. Evaluation results (2026-09-26, Groq free tier)

| Run | Served by | Catalog precision | Catalog recall | Exact match | p50 / p95 |
|---|---|---|---|---|---|
| Dev set, first live run (68 queries) | mostly qwen; gpt-oss after qwen's daily cap | 0.867 | 1.000 | 0.68 | 519 / 1602 ms |
| Same saved outputs, current merge code (`--mode rescore`) | same | 0.966 | 1.000 | 0.91 | same |
| Held-out set, live (20 new queries, never used for tuning) | 83% gpt-oss fallback (qwen daily cap), 11% rules | 0.878 | 0.837 | 0.60 | 656 / 1807 ms |
| Rules only, dev / held-out | rule_based.py | 0.958 / 0.947 | 0.929 / 0.837 | 0.78 / 0.65 | 1 / 3 ms |

- **Dev numbers are optimistic.** The grounding rules and the aliases were
  written while looking at the dev set.
- **The held-out run is the honest number,** but it mostly measures the
  fallback model. On the held-out set, every value grounding dropped was a
  correct drop. The misses were the fallback model returning empty lists, plus
  one call answered by the rules alone.
- **Re-run the held-out set on the primary model** once qwen's daily token
  cap resets, or on a paid tier:

  ```
  python -u -m scripts.eval_extractor --eval-file tests/eval_holdout.jsonl
  ```

## 8. Known limits

- **Prompt size.** The ingredients prompt is about 1,410 tokens with tiktoken
  and 1,487 as Groq counts it, against the plan's estimate of 1,185. The
  236-name allowed list alone is about 1,150 tokens. Dropping the list from
  the prompt, and keeping it only as a schema enum, was tested live: Groq
  does keep values on-list, but mapping got worse, with Redensyl missed and a
  "Sodium Hydroxide" exclusion invented. So the list stays in the prompt.
  `EXTRACTOR_ALLOWED_IN_PROMPT=false` removes it for a provider with better
  constrained decoding. Every other prompt's text is within 10% of its budget
  offline. Groq adds about 40 tokens of chat template to each call.
- **Rate limits.** The free Groq tier allows 7,000 input tokens per minute,
  200,000 tokens per day and 1,000 requests per day per model. One query costs about 4,400 input
  tokens across 10 requests, so this tier sustains about 1.5 queries per
  minute and about 45 queries per day on the primary model. Production needs the Dev tier or a different provider. A 429 is
  handled like any other error: fallback model, then rules.
- **Caching.** The cache is per process. Use a shared store when running
  several workers.
