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

The retrieval agent (section 10) implements this contract. It extends the
relaxation order, and replaces `matched_handles` with its own name search.


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

## 9. Pre-qualification stage (rewriter + router)

Every chat message now passes through a pre-qualification stage before the
extractor. Two small LLM calls run in parallel with `asyncio.gather`:

| Call | Output | Used for |
|---|---|---|
| Rewriter | `query_en`, `language`, `is_follow_up` | One standalone English request, built from the message plus the chat history. This is the only text the extractor sees. |
| Router | `route`, `needs_retrieval`, `intent`, `persona` | Decides whether retrieval runs, and whether the responder or a templated product list ends the turn. |

### Where it plugs in

```
handle_message(session_id, message)                 agents/orchestrator/orchestrator.py
  SessionStore.get(session_id) -> history, last_products     agents/prequal/session_context.py
  prequalify(message, ctx)  -- rewriter || router            agents/prequal/agent.py
  needs_retrieval?
    no  -> respond(ctx)                                       agents/responder/responder.py (stub)
    yes -> extract_filters(query_en)   (no context)           agents/filter_extractor/agent.py
           retrieve(filters, query_en)                        agents/retrieval/retrieval.py
           products_only and results -> templated intro + cards, no LLM
           otherwise                 -> respond(ctx + products)
  SessionStore.save(...)  -- history += (message, reply); last_products = products shown
```

### Reuse of the existing layers

- **LLM calls.** Both calls go through `FallBack.aconstrained_invoke`, with
  the same primary and fallback routes as the extractor
  (`PREQUAL_PRIMARY_ROUTE` and `PREQUAL_FALLBACK_ROUTE` default to the
  `EXTRACTOR_*` routes). They use the same per-route timeouts and chain
  deadline, and `route_kwargs` gives reasoning models extra tokens. The shared
  per-loop `httpx.AsyncClient` means both calls, and the extractor's 10,
  draw from one connection pool.
- **Structured output.** JSON-schema dicts (`agents/prequal/schemas.py`) are
  checked again in code after each call.
- **Prompts.** These live in `prompts/prequal/{rewriter,router}.md` and use
  the same `## The prompt` block loader. Everything before the trailing
  `HISTORY:` line is the static system message, a prefix that provider prompt
  caching can reuse. The history, last products and query form the human
  message.
- **Cache.** `TTLCache`, the extractor's cache class, is used twice: once as
  the prequal result cache keyed by (session_id, message), and once as the
  in-process session store.

### Session state

- **Chat history.** The drug assistant keeps chat history on the client
  (`chat_hist` in each request). This project had no session state, so
  `SessionStore` keeps it per `session_id`, behind a get/save interface that a
  Redis store can replace.
- **What is sent to the LLM.** The last 6 messages, with user messages cut to
  300 characters and assistant messages to 150, plus the numbered names of up
  to 5 products shown in the previous turn. `last_filters` is kept for
  debugging only and is never sent.
- **When `last_products` changes:**
  - It is replaced only when retrieval ran.
  - A turn without retrieval (such as "thanks") leaves it unchanged, so
    "the second one" still resolves on the next turn.

### Changes to the extractor

- **The names call no longer receives `{{context}}`.** The rewriter already
  turns "it", "ده" and "التاني" into the product's name, so `query_en`
  carries it. The orchestrator calls `extract_filters(query_en)` with no
  context. `extract_filters(query, context)` keeps its signature for
  standalone use: grounding and the rule-based fallback still read the context
  when one is passed.
- **`EXTRACTOR_TRANSLATE` is replaced by the rewriter** in the chat
  pipeline. It stays available, off by default, for standalone
  `extract_filters` use.

### Fallbacks

| Failure | Result |
|---|---|
| Rewriter, both models | `query_en` is the raw message, and `language` comes from script detection. Metadata filters are used only when the message is English and there is no history. Otherwise `skip_metadata_filters=true`, and retrieval uses text search on the raw message. |
| Router, both models | `needs_response`, `needs_retrieval=true`, `intent=other`, `persona=unknown` |
| Both | Both defaults apply, an error is logged, and the responder still answers |
| Responder raises | A fixed apology in the user's language |

Each call catches its own errors, so one failure never cancels the other.

### Retrieval and responder

The retrieval described here was a stand-in. It is replaced by the retrieval
agent in section 10.


- **Retrieval.** The project had no retrieval layer. `agents/retrieval`
  applies the section 5 contract to the in-memory catalog:
  - AND across keys, OR within a key.
  - Named products (`matched_handles`) take priority.
  - Filters relax in the stated order, and `exclude` is never relaxed.
  - Unmatched terms act as soft boosts or penalties.
  - A token-overlap text search is used when there are no usable filters.
    This is the "semantic fallback"; no embedding model is configured in this
    project.
- **Responder.** The responder is out of scope. `respond(ctx) -> str` defines
  the interface (`ResponderContext`) and returns a templated placeholder
  reply. It is marked TODO.

### Speculative extraction

With `PREQUAL_SPECULATIVE_EXTRACTOR=true`, the extractor starts as soon as the
rewriter returns, without waiting for the router. If the router then says
`needs_retrieval=false`, the extractor task is cancelled. The flag is off by
default, because a cancelled extraction still spends its 10 calls' tokens.

### Deviations from the plan, and why

- **New layers.** Retrieval, a responder stub, an in-process session store
  and the orchestrator were added, because the project had none of them.
- **Rewriter fallback.** Metadata filters are kept only when the raw message
  is English and there is no history. The plan says "mostly Latin script",
  but Arabizi is Latin script too, and the extractor expects English.
- **Cache key.** The plan keys the cache on (session_id, message). The key
  also includes the history the prompt sees, so the same words later in the
  chat ("any cheaper?") don't get a stale rewrite. For a retry of a turn that
  already completed, the trailing copy of the message is removed from the
  history before the key is built, so retries still hit the cache.
- **`last_products`** changes only on turns where retrieval ran (see
  "Session state").
- **Prompt changes after the first live eval:**
  - The router's `route`, `needs_retrieval`, `intent` and `persona` lines were
    made explicit. The first run showed:
    - `needs_retrieval` read as "needs a product search"
    - "do you have …?", "cheaper?" and "for <skin type>?" routed to
      `needs_response`
    - "يا دكتور" (addressing the assistant) and "my son" read as the doctor
      persona
  - The rewriter got one line telling it to translate small talk rather than
    return an empty `query_en`.
  - Both changes are noted in the prompt files.
- **Route from intent** (`PREQUAL_ROUTE_FROM_INTENT`, on by default). When
  the router says `find_products` or `refine_products` but `needs_response`,
  the route becomes `products_only`.
  - In the live eval, intent was more accurate than route, and all 4 such
    contradictions were find requests sent to `needs_response`.
  - The risk is a mixed "find + question" message tagged `find_products`.
    None of the 62 eval conversations did this, and the prompt tells the model
    to use a question intent for those.
- **`warm_up()`.** The first live call took 4.2 s. That time was synchronous
  cold-start work that ran inside the calls' timeouts: model construction,
  catalog and template loading. The orchestrator's `warm_up()` does it at
  startup.
- **Prompt size.** Counts come from Groq's `usage`:

  | Prompt | Plan estimate | qwen, before tuning | gpt-oss-120b, after tuning |
  |---|---|---|---|
  | Rewriter | ~450 | 666 | 786 |
  | Router | ~400 | 478 | 739 |

  The Arabic examples tokenize heavily. The offline tiktoken counts are 594
  and 496.
- **Extractor eval rows.** q57, q58 and h18 in the extractor eval sets were
  rewritten into the resolved form the rewriter now produces, because the
  names call no longer receives context.

### Evaluation (2026-09-26, Groq free tier)

- **Eval sets.**
  - `tests/prequal_eval.jsonl` is the dev set: 48 conversations.
  - `tests/prequal_holdout.jsonl` has 14 conversations, written before the
    router prompt was tuned.
- **Running it.** `python -m scripts.eval_prequal` runs the live prequal
  stage, then the extractor on `query_en`. The extractor uses the offline
  rules by default; `--extractor llm` uses the LLM.

| Run | Served by | Route | needs_retrieval | Intent | Rewrite check | E2E filter F1 (rules extractor) | p50 / p95 |
|---|---|---|---|---|---|---|---|
| Configured routes (qwen → gpt-oss-20b), dev | 40 of 48 rewrites and 41 of 48 routes fell back to the safe default: both models had hit their 200k-tokens-per-day cap | 0.604 | 0.854 | 0.146 | 0.354 | 0.556 | 264 / 683 ms |
| gpt-oss-120b, untuned prompts, dev | model (2 empty rewrites, 1 timeout) | 0.896 | 0.812 | 0.854 | 0.938 | 0.898 | 647 / 1262 ms |
| gpt-oss-120b, tuned prompts, dev | model (1 timeout) | 0.875 | **0.979** | 0.917 | 0.938 | **0.880** | 780 / 1517 ms |
| Same run, with route from intent (`--mode rescore`) | same | 0.938 | **0.979** | 0.917 | 0.938 | **0.880** | same |
| gpt-oss-120b, tuned, **holdout**, with route from intent | model | **1.000** | **1.000** | 1.000 | 0.929 | 0.791 | 617 / 1210 ms |

How to read these results:

- **The first row measures the fallbacks, not the models.** It shows the
  "both calls fail" path: every turn still got a routable result, in about
  260 ms.
- **gpt-oss-120b is a stand-in.** The configured primary (qwen) was out of
  daily quota, so the runs used the only other structured-output model on
  this account, the drug assistant's own main model. It was run as the only
  route (`PREQUAL_PRIMARY_ROUTE=groq:openai/gpt-oss-120b`,
  `PREQUAL_USE_FALLBACK_MODEL=false`). **These accuracy and latency numbers
  are not qwen's.** qwen, measured on 6 warm calls, answered in 336 to 592 ms.
- **Route target not met on dev.** Dev reaches 0.938 against 0.95. The
  misses:
  - c15: "do you have a sunscreen?" → `product_info`
  - c48: "cheaper than X" → `price_offer`
  - c22: a timeout that fell back to the default; with the fallback model on,
    it would probably have been answered
  - Holdout is 14/14.
- **The dev numbers are optimistic.** The router prompt was tuned while
  looking at the dev set.
- **Rewriter quality.** References resolve 7/7 and topic changes 5/5, with
  no old constraints carried over. The misses are Arabizi vocabulary:
  "bo2a3" (stains) became "rash", "shafayef" (lips) became "eyelashes", and
  "خشونة" (osteoarthritis) became "rough skin".
- **The E2E F1 uses the rule-based extractor,** because the LLM extractor
  costs about 7k tokens per row. The holdout misses (0.791) are mostly
  category false positives from the rules, for example "Skin Care" for an
  intimate wash. The LLM extractor was verified live on the motivating
  conversation only: `product_type=hair serum`, `exclude=Silicone`.
- **Latency targets not met on gpt-oss-120b.** p50 must be ≤ 0.7 s and
  p95 ≤ 1.2 s. gpt-oss-120b is a reasoning model, and these targets were set
  for the fast tier.
- **Still to verify.** Re-run both sets on the configured routes once qwen's
  daily cap resets, or on a paid tier:

  ```
  python -u -m scripts.eval_prequal
  python -u -m scripts.eval_prequal --eval-file tests/prequal_holdout.jsonl
  python -u -m scripts.eval_prequal --extractor llm      # full products_only path latency
  ```

  One full turn uses about 7k input tokens (prequal about 1.4k, the extractor
  about 6k), which is nearly the free tier's 7,000 per minute.

## 10. Retrieval agent (exact filters + name BM25 + semantic search)

Retrieval picks the top `k` products for one request. It uses no LLM: it
combines three deterministic signals over the in-memory catalog, and replaces
the stand-in `agents/retrieval/retrieval.py` described in section 9.

| Signal | Input | Method |
|---|---|---|
| Exact filters | the extractor's catalog keys | inverted index per key, set intersection |
| Name match | the extractor's `name_en` / `name_ar` | BM25 over character bigrams, plus a Dice check |
| Semantic | the rewriter's `query_en` | cosine similarity against each product's `description` embedding |

### Where it runs

```
prequalify(message)  -- rewriter || router  ->  query_en, route, needs_retrieval, intent, k
  needs_retrieval = true:
    asyncio.gather(extract_filters(query_en), semantic_search(query_en))   -> filters, sem
    retrieve(filters, sem, query_en, intent, k)                            -> RetrievalResult
      products_only, products found, nothing relaxed  -> templated intro + cards (final, no LLM)
      otherwise                                       -> respond(ctx + the whole RetrievalResult)
```

- The semantic search runs in parallel with the extractor. With
  `PREQUAL_SPECULATIVE_EXTRACTOR=true` both start when the rewriter returns,
  and both are cancelled if the router then says `needs_retrieval=false`.
- When the rewriter failed (`skip_metadata_filters`), the extractor is
  skipped. The semantic search still runs on the raw message.

### Module layout

```
agents/retrieval/
  index_builder.py  product pool, inverted indexes, name indexes, embedding text + disk cache
  name_search.py    name normalization (en/ar), bigram tokenizer, BM25 + Dice acceptance
  semantic.py       product embedding matrix, query embedding (timeout, LRU), cosine scores
  filters.py        exact filter step: OR within a key, AND across keys, relaxation, bundle rule
  fusion.py         scores the candidate set: semantic + boosts, tie-breakers
  agent.py          retrieve(filters, sem, query_en, intent, k) -> RetrievalResult
models/retrieval.py RetrievedProduct, RetrievalResult
llm/embeddings.py   embedding client factory ("router:model" routes, like llm_models.py)
```

### Embedding client (proposal)

Neither this project nor the drug assistant has an embedding client or a
vector store, and the only configured provider (Groq) has no embeddings
endpoint. So:

- **`llm/embeddings.py`** builds a LangChain `Embeddings` object from a
  route, the same way `llm_models.py` builds chat models:
  - `hf:<model>`: a local sentence-transformers model. This needs no key.
  - `gpt:<model>`, `gemini:<model>` and `ollama:<model>`: those providers'
    LangChain embedding classes, whose packages are already in
    `requirements.txt`.
- **Default: `hf:sentence-transformers/all-MiniLM-L6-v2`.** It is small
  (22M parameters, 384 dimensions), English, and runs on CPU in a few
  milliseconds per query. `RETRIEVAL_EMBEDDING_ROUTE` switches the model with
  no code changes.
- **No vector database.** 108 products fit in one numpy matrix.
- **Imports are lazy.** Only the semantic index loads `sentence_transformers`
  (and so torch). The unit tests use a fake embedder, so they stay offline.

### Model and embedding caching (taken from the drug assistant)

| Drug-assistant piece | What it does there | Here |
|---|---|---|
| `model_manager.py` | Local-cache-first spaCy install. A `.download_complete` marker under `models/<name>/` is trusted only with a live check that the model is installed; the download runs only when missing | `model_manager.py::ensure_hf_model_downloaded`, detailed below |
| `agents/meta_data_fiter/engine_registry.py` | Search index built once per process, behind an `_UNSET` singleton. One pickle, `cache/<name>_v{_SCHEMA_VERSION}_{sha256(data)[:16]}`. Stale files are deleted on rebuild, a corrupt file rebuilds, and `warm_up_*()` runs in the lifespan | `agents/retrieval/embedding_registry.py`, detailed below |
| `agents/image_pii/helpers.py` | Slow engine init in a background daemon thread, started on first use, with a bounded wait per call. Not ready means that call skips it without giving up; a failure is cached | `agents/retrieval/semantic.py::_ModelLoader`, detailed below |

**`model_manager.py`**

- **Where the model lives.** An `hf:` model is downloaded once into
  `RETRIEVAL_EMBEDDING_MODEL_DIR/<repo slug>/` and loaded from that path.
  sentence-transformers makes no Hub call on restart, and loading works
  offline; this was checked with `HF_HUB_OFFLINE=1`.
- **The marker.** It counts only if `config.json` and the weights are also
  present.
- **What is downloaded.** ONNX, OpenVINO, TF, Rust and duplicate `.bin`
  weights are skipped. The default repo is about 1 GB in full, and 88 MB of it
  is what sentence-transformers loads.

**`embedding_registry.py`**

- **One file.** The product matrix is saved as
  `cache/product_embeddings_v{_SCHEMA_VERSION}_{model}_{sha256(catalog)[:16]}.npz`.
  `np.savez`/`np.load(allow_pickle=False)` is used instead of pickle, since
  the data is just two arrays.
- **Bump `_SCHEMA_VERSION` when `embedding_text()` changes.** It sits next to
  that function for this reason: the key hashes only the data, so without the
  version a code change would load stale vectors.

**`semantic.py::_ModelLoader`**

- **Background load.** The model and matrix load in a daemon thread. Each
  request waits at most its semantic timeout (1.5 s); if the model isn't
  ready, that request gets status `loading` (fallback `no_semantic`) and a
  later request uses the finished load.
- **Failures are permanent.** A failed load is cached until restart (status
  `failed`), as in the drug assistant.
- **Warm-up.** `warm_up()` waits up to `RETRIEVAL_MODEL_INIT_TIMEOUT_S` (60 s).
- **Switch.** `RETRIEVAL_SEMANTIC_ENABLED=false` turns semantic search off,
  like `ENABLE_IMAGE_PII`: no model loads.

**Startup cost on this machine** (shared with other sessions, so slower than
an idle host):

- Cold first start: 45 s (download plus torch import).
- Later starts: 18–40 s, almost all of it the torch import.
- The product matrix loads from its file in milliseconds.

### Index (built once per process)

- **Product pool.** `products` only. Bundles keep `product_kind = bundle`,
  and are also indexed under each of their `contains_product_types`.
- **Inverted indexes.** Filter key → value → set of handles, following
  `field_map`: `hero_ingredient` is `hero_ingredient.name`, and `ingredients`
  is `ingredients_canonical`.
- **Two name indexes.** `rank_bm25.BM25Okapi` over character bigrams with
  boundary markers, one index for `name` and one for `name_ar`.
  - Normalization strips sizes (`120ml`, `50 gm`, `30 مل`), symbols
    (`®`, `™`, `×`) and punctuation.
  - Arabic folds أ/إ/آ → ا, ة → ه, ى → ي and the four "v" letters
    (ڤ ڨ ڈ ڄ) → ف, and drops tashkeel and tatweel.
  - `get_scores` loops over every name in Python for each query bigram, so
    each bigram's per-name score vector is precomputed from the fitted
    model. A name query then costs microseconds instead of about 1.3 ms, and
    the scores are identical to the library's: the largest difference over
    all 1,063 typo queries was 3.6e-14.

### The three steps

- **Exact filters** (`filters.py`):
  - OR within a key, AND across keys, `ingredients.include` all-of,
    `ingredients.exclude` any-of. Exclude is applied in every fallback.
  - On an empty candidate set, keys are dropped one at a time:
    `product_form → suitable_for → concerns → product_group → category →
    hero_ingredient → ingredients.include → product_type → brand`.
  - With every key dropped: the pool minus excluded products, and
    `fallback = "semantic_only"`.
  - Bundle rule: bundles stay only for product_type `bundle`, group
    `Bundles & Offers`, a bundle matched by name, or intent `price_offer`.
- **Names** (`name_search.py`):
  - A hit needs BM25 ≥ 0.8 × the top score for that queried name and bigram
    Dice ≥ 0.35. Up to 10 hits are kept per name.
  - `name_en[i]` and `name_ar[i]` are the same name in two scripts. A name is
    unresolved only when neither form hits.
  - Queries are routed by script, so Arabic text in `name_en` still reaches
    the Arabic index.
- **Semantic + fusion** (`semantic.py`, `fusion.py`):
  - `final = cosine + boosts`, with the weights in `RETRIEVAL_FUSION_WEIGHTS`.
  - Out-of-stock products always rank below in-stock ones.
  - Tie-breakers: name score, then lower price, then handle.
  - The query embedding has a 1.5 s timeout and an LRU cache (`TTLCache`).

### Output contract and hand-off

`RetrievalResult` (`models/retrieval.py`) follows the plan's contract, with
two additions:

- `RetrievedProduct.url_ar`, which the old cards had.
- `meta`:
  - `latency_ms`
  - `semantic` (the status)
  - `fallbacks`, listing both fallbacks when both happen; `fallback` holds the
    first, and `no_semantic` wins.
  - `relaxed_values`, the requested values of each relaxed key, so the
    responder can say "no spray found, here are gels"
  - `bundles_allowed`
  - `excluded_count`
  - `filters_skipped`

Orchestrator hand-off:

| Case | Result |
|---|---|
| `products_only`, products found, nothing relaxed | Templated intro + cards. Final. |
| `products_only`, nothing found | Responder, reason `no_results` |
| `products_only`, some keys relaxed | Responder, reason `relaxed` |
| `needs_response` | Responder, with the whole `RetrievalResult` in `ResponderContext.retrieval` |

### Router change (`k`)

- **Schema.** `"k": {"type": ["integer","null"], "minimum": 1, "maximum": 20}`
  is required. Groq's strict mode accepts it; both configured routes were
  checked live before the change.
- **Validation.** A bad `k` never fails the router call, because route and
  intent matter more:
  - null, 0, negative or non-numeric → null (the default applies)
  - above 20 → 20
  - `"4"` or `4.0` → 4
- **`PrequalResult.k`.** The router's `k`, else `RETRIEVAL_K_DEFAULT` (10),
  capped at `RETRIEVAL_K_MAX` (20). `meta.k_source` is `router` or `default`.
- **Prompt.** The plan's rule line and Example 3 are in verbatim. Examples 1
  and 2 gained `"k":null`.
  - One clause was added after the first live run: `"something" (7aga,
    حاجة) is not a count`. The router had read "3ayez 7aga lel 2eshra" as
    k=1, which would show a single product.
  - The prompt is now 630 tiktoken tokens, up from ~496; its budget test is
    now 650.

### Deviations from the plan, and why

- **Location.** The plan suggested a top-level `retrieval/` package. The
  modules live in `agents/retrieval/` and `models/retrieval.py`, following
  this project's layout. The stand-in `agents/retrieval/retrieval.py` was
  removed.
- **Caching follows the drug assistant** (see "Model and embedding caching"
  above). The plan's three files (`embeddings.npy`, `handles.json`, model
  name) became one versioned `.npz`, with the model and the catalog hash in
  its name.
- **`matched_handles` is no longer used by retrieval.** The plan's name step
  (bigram BM25 on `name_en` / `name_ar`) replaces it. The extractor still
  fills it, and its eval still scores it.
- **Unmatched concerns boost too.** `unmatched.concerns` found in the
  description adds `unmatched_concern` (0.05). The plan's formula lists only
  unmatched include terms, but the extractor's hand-off contract (section 5)
  says unmatched concerns boost.
- **More free-of phrasings.** Besides "<term>-free" and "free from <term>",
  these also count: "free of", "contains no", "without" (within one
  sentence, so lists like "free from ammonia, aluminum, and alcohol" match),
  and "no / 0% <term>". Plurals match both ways ("parabens" finds
  "paraben").
- **Precision@k** is the relevant share of the returned list, not of `k`
  slots. **Recall@k** divides by `min(|relevant|, k)`. Otherwise a request
  with 2 relevant products and k=10 could never exceed 0.2 precision, and a
  k=3 request for 6 relevant products could never reach recall 1.0.

### Dice threshold sweep (a recommendation, not applied)

The plan's Dice floor, 0.35, lets through long names that share only common
bigrams with the query: "Nonexistent Thing" matches a Vacation deodorant at
exactly 0.35. The sweep used the 1,063 generated typo queries, plus 36 names
that are not in the catalog (competitor products, invented Majestic names).

| Dice floor | Typo top-1 | Typo top-3 | False accepts (of 36) |
|---|---|---|---|
| 0.35 (plan, default) | 0.956 | 0.994 | 23 |
| 0.40 | 0.957 | 0.994 | 20 |
| 0.45 | 0.959 | 0.995 | 14 |
| 0.50 | 0.962 | 0.994 | 13 |

- **A higher floor costs no typo recall.** Top-1 even improves, because noisy
  accepts that outranked the right product disappear.
- **Some false accepts can't be fixed with bigrams.** The ones left at 0.5
  are brand plus real words ("Vacation Retinol Night Cream" → Uni-White
  Night Cream).
- **Risk is lower in practice.** The extractor's grounding already drops
  names without a Majestic brand or product line, so competitor names rarely
  reach retrieval.
- **Recommendation:** `RETRIEVAL_NAME_DICE_MIN=0.45`. It was left at the
  plan's 0.35 pending a decision.

### Evaluation (2026-09-26)

**Retrieval** (`python -m scripts.eval_retrieval`, 56 cases in
`tests/retrieval_eval.jsonl`, no LLM). Relevance was labelled by judgment,
not by what the filters return, so exact-match limits show up as misses.

| Run | precision@k | recall@k | MRR | Forbidden products returned |
|---|---|---|---|---|
| Semantic on (all-MiniLM-L6-v2) | 0.874 | 0.949 | 0.960 | 0 rows |
| Semantic off (`--no-semantic`) | 0.865 | 0.935 | 0.955 | 0 rows |

- **The plan's cases:**
  - hair serum without silicone: exact
  - sunscreen spray for oily skin: form relaxed, recall 0.75 (see limits)
  - pregnant, no retinol or salicylic acid: exact, nothing forbidden
  - compare, deodorant line and k=3: all perfect
- **Semantic adds little where filters exist** (+0.014 recall). It matters
  in the broad and filter-less rows.
- **Misses are exact-match limits, not bugs:**
  - r02: the "all skin types" spray isn't labeled oily skin
  - r21: only one balm is labeled "dry lips"
  - r22: the caffeine serum is typed eye serum, not eye cream
  - r56: a wrongly extracted suitable_for makes relaxation drop product_form
    first
  - r47: raw Arabic with no filters, when the rewriter failed. The English
    model scores 0 there.

**Latency** (offline):

| Measure | p50 | p95 | Target |
|---|---|---|---|
| Filters + names + fusion | 0.34 ms | 1.9 ms | ≤ 10 ms |
| Semantic search (uncached query embedding, CPU) | 21 ms | 35 ms | ≤ 300 ms |

Before the precomputed BM25 vectors, the retrieval p95 was 9.0 ms.

**Name typo test** (`python -m scripts.name_typos`):

- 1,063 generated queries: English and Arabic, 5 variants per name (delete,
  swap, replace, 2 edits, brand dropped).
- **top-1 0.956** (target ≥ 0.95) and **top-3 0.994** (target ≥ 0.99).
- All 5 real misspellings from the plan resolve.
- The misses:
  - bundles whose names contain the single product's name, such as
    "... Serum + 1000 Laser Pulses Card"
  - very short names with 2 edits

**Router `k`** (live, configured primary qwen/qwen3.8-27b, 7 new
`count`-tagged rows in `tests/prequal_eval.jsonl`):

- 7/7 after the "something" clause (6/7 before):
  - "show me 3", "أحسن ٢", "top five", "تلاتة" and "2 lip balm" are read as
    counts.
  - "wa7ed bas" → 1.
  - "3ayez 7aga" → null.
- A full dev-set regression run of the router was started. It was stopped
  after 8 rows because qwen hit the account's 200k tokens-per-day cap, so the
  run would have measured the fallback model.
- **Still to do:** re-run `python -u -m scripts.eval_prequal` once the cap
  resets, to confirm route and intent accuracy with the new k line.

**End-to-end smoke test** (`python -m scripts.smoke_e2e`, live): six raw
Arabic, Arabizi and English messages, each through prequal → extractor ∥
semantic → retrieval.

- **6/6 turns as expected:** right k, an expected product in the top 3,
  nothing forbidden.
- **The products_only path** (prequal + extractor ∥ semantic + retrieval):
  1.56–2.37 s, **p95 2.37 s** against a 3 s target.
- **Caveat: these times are pessimistic.** qwen was over its daily cap, so
  most calls went to gpt-oss-20b after a 429 round trip (118 rate-limit
  errors in the log).
- **Semantic always finished inside the extractor's time,** so it added no
  wall-clock time. It took 100–680 ms, not 35 ms, while the extractor's 10
  calls ran.
- **Retrieval measured 0.6–26 ms of wall time inside the live pipeline,**
  against 0.3–3 ms of isolated compute. Timing retrieval directly after an
  embedding call gave the same median (0.67 ms), so the spikes look like
  machine load and GC pauses. Six sessions share this machine.
- **The rewriter's Arabizi misreadings** ("3ara2" sweat → "oily skin",
  "2eshra" dandruff → "face") show up downstream as relaxed filters. That is
  the rewriter limit noted in section 9, not a retrieval one.

### Known limits

- **"all skin types" does not satisfy a specific skin type.** Exact match
  follows the plan, so the Sunscreen Spray (all skin types) is missing from
  "sunscreen spray for oily skin". Treating `all skin types` as matching
  every skin-type value would fix r02 and similar cases. It's a one-line
  change in `filters.key_matches`.
- **Relaxation drops by key order, not by key confidence.** A wrong
  suitable_for is dropped only after product_form, as in r56.
- **The embedding model is English.** The rewriter-failed path embeds the raw
  message, so Arabic or Arabizi text scores near zero there. A multilingual
  model (for example `paraphrase-multilingual-MiniLM-L12-v2`, via
  `RETRIEVAL_EMBEDDING_ROUTE`) would help that path only.
- **The session remembers 5 of the 10 products shown**
  (`PREQUAL_LAST_PRODUCTS`), so "the seventh one" can't be resolved.
- **A failed model load is not retried until restart**, as in the drug
  assistant. For example, a network error during the first download leaves
  semantic search off until the next start.
- **The caches are per host.** Each worker keeps its own `models/embeddings/`
  and `cache/` copy.
