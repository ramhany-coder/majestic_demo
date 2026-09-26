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
