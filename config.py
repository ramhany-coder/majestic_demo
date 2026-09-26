import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


class Settings:
    GEMINI_API = os.getenv("GEMINI_API")
    GROQ_API = os.getenv("GROQ_API")
    OLLAMA_PATH = os.getenv("OLLAMA_PATH", "http://localhost:11434")
    GPT_API = os.getenv("GPT_API")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    # Z.ai (GLM) serves every LLM call by default, through its OpenAI-compatible
    # endpoint (use https://open.bigmodel.cn/api/paas/v4/ for a BigModel key).
    ZAI_API_KEY = os.getenv("ZAI_API_KEY")
    ZAI_BASE_URL = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/")

    # --- Metadata filter extractor (agents/filter_extractor) -----------------
    # Routes are "router:model" (router is one of llm.helpers.Helpers.routers_list).
    # The primary is the smallest fast model this project can reach; the
    # fallback is tried once per call when the primary errors or times out.
    EXTRACTOR_PRIMARY_ROUTE = os.getenv("EXTRACTOR_PRIMARY_ROUTE", "zai:glm-5.3-flash")
    EXTRACTOR_FALLBACK_ROUTE = os.getenv("EXTRACTOR_FALLBACK_ROUTE", "zai:glm-5.3-flash")
    # Feature flags for the two fallback stages.
    EXTRACTOR_USE_FALLBACK_MODEL = _env_bool("EXTRACTOR_USE_FALLBACK_MODEL", True)
    EXTRACTOR_USE_RULE_FALLBACK = _env_bool("EXTRACTOR_USE_RULE_FALLBACK", True)
    # Per-attempt timeouts (seconds), and a hard per-call deadline over both attempts.
    # Sized for glm-5.3-flash: about 2-3 s per call even for a few output tokens,
    # with the odd call hanging far longer (the retry covers those).
    EXTRACTOR_TIMEOUT_S = _env_float("EXTRACTOR_TIMEOUT_S", 4.0)
    EXTRACTOR_FALLBACK_TIMEOUT_S = _env_float("EXTRACTOR_FALLBACK_TIMEOUT_S", 4.0)
    EXTRACTOR_CALL_DEADLINE_S = _env_float("EXTRACTOR_CALL_DEADLINE_S", 8.0)
    # Reasoning models (gpt-oss, glm-5.x -- its thinking cannot be turned off) spend completion tokens thinking before they
    # answer; this many tokens are added to max_tokens when a call is routed to one.
    EXTRACTOR_REASONING_TOKEN_ALLOWANCE = int(os.getenv("EXTRACTOR_REASONING_TOKEN_ALLOWANCE", 300))
    # Optional rewrite-to-English call before the fan-out (Arabic / Arabizi users).
    EXTRACTOR_TRANSLATE = _env_bool("EXTRACTOR_TRANSLATE", False)
    EXTRACTOR_TRANSLATE_MAX_TOKENS = int(os.getenv("EXTRACTOR_TRANSLATE_MAX_TOKENS", 150))
    # Keep the ALLOWED line in each prompt. Turn off only for a provider that
    # enforces schema enums by constrained decoding (the schema alone is then enough).
    EXTRACTOR_ALLOWED_IN_PROMPT = _env_bool("EXTRACTOR_ALLOWED_IN_PROMPT", True)
    EXTRACTOR_MAX_QUERY_CHARS = int(os.getenv("EXTRACTOR_MAX_QUERY_CHARS", 500))
    EXTRACTOR_CACHE_TTL_S = _env_float("EXTRACTOR_CACHE_TTL_S", 24 * 3600)
    EXTRACTOR_CACHE_SIZE = int(os.getenv("EXTRACTOR_CACHE_SIZE", 2048))
    # Per-call max_tokens. Override any subset with JSON, for example
    # EXTRACTOR_MAX_TOKENS='{"ingredients": 120}'.
    EXTRACTOR_MAX_TOKENS = {
        "names": 120,
        "brand": 30,
        "category": 30,
        "product_group": 40,
        "product_type": 40,
        "product_form": 30,
        "concerns": 50,
        "suitable_for": 40,
        "hero_ingredient": 40,
        "ingredients": 100,
        **json.loads(os.getenv("EXTRACTOR_MAX_TOKENS", "{}")),
    }
    # Per-call primary timeout overrides, same JSON form (default EXTRACTOR_TIMEOUT_S).
    EXTRACTOR_CALL_TIMEOUTS = json.loads(os.getenv("EXTRACTOR_CALL_TIMEOUTS", "{}"))

    # --- Pre-qualification stage (agents/prequal): rewriter + router ---------
    # Same fast tier as the extractor unless overridden.
    PREQUAL_PRIMARY_ROUTE = os.getenv("PREQUAL_PRIMARY_ROUTE", EXTRACTOR_PRIMARY_ROUTE)
    PREQUAL_FALLBACK_ROUTE = os.getenv("PREQUAL_FALLBACK_ROUTE", EXTRACTOR_FALLBACK_ROUTE)
    PREQUAL_USE_FALLBACK_MODEL = _env_bool("PREQUAL_USE_FALLBACK_MODEL", True)
    PREQUAL_REWRITER_MAX_TOKENS = int(os.getenv("PREQUAL_REWRITER_MAX_TOKENS", 120))
    PREQUAL_ROUTER_MAX_TOKENS = int(os.getenv("PREQUAL_ROUTER_MAX_TOKENS", 40))
    # Primary-attempt timeouts, the fallback attempt's timeout, and a hard
    # deadline per call over both attempts (seconds).
    PREQUAL_REWRITER_TIMEOUT_S = _env_float("PREQUAL_REWRITER_TIMEOUT_S", 4.0)
    PREQUAL_ROUTER_TIMEOUT_S = _env_float("PREQUAL_ROUTER_TIMEOUT_S", 4.0)
    PREQUAL_FALLBACK_TIMEOUT_S = _env_float("PREQUAL_FALLBACK_TIMEOUT_S", 4.0)
    PREQUAL_REWRITER_DEADLINE_S = _env_float("PREQUAL_REWRITER_DEADLINE_S", 8.0)
    PREQUAL_ROUTER_DEADLINE_S = _env_float("PREQUAL_ROUTER_DEADLINE_S", 8.0)
    # Chat context sent to both calls.
    PREQUAL_HISTORY_MESSAGES = int(os.getenv("PREQUAL_HISTORY_MESSAGES", 6))
    PREQUAL_HISTORY_USER_CHARS = int(os.getenv("PREQUAL_HISTORY_USER_CHARS", 300))
    PREQUAL_HISTORY_ASSISTANT_CHARS = int(os.getenv("PREQUAL_HISTORY_ASSISTANT_CHARS", 150))
    PREQUAL_LAST_PRODUCTS = int(os.getenv("PREQUAL_LAST_PRODUCTS", 5))
    PREQUAL_MAX_QUERY_CHARS = int(os.getenv("PREQUAL_MAX_QUERY_CHARS", 500))
    # (session_id, message) cache, so a retried request is not billed twice.
    PREQUAL_CACHE_TTL_S = _env_float("PREQUAL_CACHE_TTL_S", 300)
    PREQUAL_CACHE_SIZE = int(os.getenv("PREQUAL_CACHE_SIZE", 4096))
    # Start the extractor as soon as the rewriter returns; cancel it if the
    # router then says no retrieval is needed.
    PREQUAL_SPECULATIVE_EXTRACTOR = _env_bool("PREQUAL_SPECULATIVE_EXTRACTOR", False)
    # Router output with intent find_products/refine_products but route
    # needs_response is corrected to products_only (see agents/prequal/schemas.py).
    PREQUAL_ROUTE_FROM_INTENT = _env_bool("PREQUAL_ROUTE_FROM_INTENT", True)
    # Greetings / thanks / goodbyes end the turn in prequal with a templated
    # reply: a message made only of those words skips both LLM calls, and a
    # router intent=greeting skips everything after prequal.
    PREQUAL_SMALL_TALK_END = _env_bool("PREQUAL_SMALL_TALK_END", True)

    # --- Retrieval (agents/retrieval): exact filters + name BM25 + semantic ---
    # Products returned per turn: the router's k, else the default; always capped.
    RETRIEVAL_K_DEFAULT = int(os.getenv("RETRIEVAL_K_DEFAULT", 10))
    RETRIEVAL_K_MAX = int(os.getenv("RETRIEVAL_K_MAX", 20))
    # Name search: BM25 over character bigrams. b is low because names are short.
    RETRIEVAL_BM25_K1 = _env_float("RETRIEVAL_BM25_K1", 1.2)
    RETRIEVAL_BM25_B = _env_float("RETRIEVAL_BM25_B", 0.3)
    # A name hit needs BM25 >= this share of the top score for that queried name,
    # and a bigram Dice coefficient >= RETRIEVAL_NAME_DICE_MIN. The plan's 0.35
    # let long unrelated names through; 0.45 rejects more of them at no typo-recall
    # cost (ARCHITECTURE_NOTES.md section 10, "Dice threshold sweep").
    RETRIEVAL_NAME_REL_SCORE = _env_float("RETRIEVAL_NAME_REL_SCORE", 0.8)
    RETRIEVAL_NAME_DICE_MIN = _env_float("RETRIEVAL_NAME_DICE_MIN", 0.45)
    RETRIEVAL_NAME_MAX_HITS = int(os.getenv("RETRIEVAL_NAME_MAX_HITS", 10))
    # Intents whose result is filled from the filtered candidates after the
    # name hits. Every other intent returns the name hits alone (when there are any).
    RETRIEVAL_FILL_INTENTS = tuple(
        i.strip() for i in os.getenv("RETRIEVAL_FILL_INTENTS", "find_products,refine_products").split(",") if i.strip()
    )
    # Drop bundles from the candidates unless the request asks for them
    # (product_type bundle, group Bundles & Offers, a named bundle, or a price/offer intent).
    RETRIEVAL_BUNDLE_RULE = _env_bool("RETRIEVAL_BUNDLE_RULE", True)
    # A suitable_for filter naming a specific skin type (oily, dry, ...) also
    # matches products labeled "all skin types". Off: exact labels only (the plan).
    RETRIEVAL_ALL_SKIN_TYPES_MATCH = _env_bool("RETRIEVAL_ALL_SKIN_TYPES_MATCH", True)
    # A product_form also matches catalog forms containing it as whole words
    # ("cream" -> "cream gel", "tinted cream", "jelly cream").
    RETRIEVAL_FORM_VARIANTS = _env_bool("RETRIEVAL_FORM_VARIANTS", True)
    # Off (default): only products matching every requested key are returned;
    # when none do, the list is empty and the responder says so. On: the plan's
    # relaxation, dropping keys in a fixed order until something matches.
    RETRIEVAL_RELAX_FILTERS = _env_bool("RETRIEVAL_RELAX_FILTERS", False)
    # Fusion: final = cosine + these boosts. Override any subset with JSON, for
    # example RETRIEVAL_FUSION_WEIGHTS='{"best_seller": 0.05}'.
    RETRIEVAL_FUSION_WEIGHTS = {
        "available": 0.05,
        "best_seller": 0.03,
        "unmatched_include": 0.05,      # per unmatched include term found in the description
        "unmatched_concern": 0.05,      # per unmatched concern found in the description
        "extra_concern": 0.03,          # per requested concern matched beyond the first
        "unmatched_exclude": -0.10,     # per unmatched exclude term mentioned in the description
        "exclude_free": 0.05,           # ... unless the description says "<term>-free" / "free from <term>"
        "bundle": -0.05,
        "all_skin_types": -0.03,        # fits a requested skin type only through "all skin types"
        **json.loads(os.getenv("RETRIEVAL_FUSION_WEIGHTS", "{}")),
    }
    # Semantic search: "router:model" (see llm/embeddings.py). The default is a
    # small local English model, so no API key is needed. Off: retrieval ranks
    # on filters, names and boosts only, and no model is loaded (low-memory hosts).
    RETRIEVAL_SEMANTIC_ENABLED = _env_bool("RETRIEVAL_SEMANTIC_ENABLED", True)
    RETRIEVAL_EMBEDDING_ROUTE = os.getenv("RETRIEVAL_EMBEDDING_ROUTE", "hf:sentence-transformers/all-MiniLM-L6-v2")
    # Where an "hf:" model is downloaded once and loaded from afterwards
    # (model_manager.py; relative paths are under the repo root). Never commit
    # it: a download-complete marker without its files forces a re-download.
    RETRIEVAL_EMBEDDING_MODEL_DIR = os.getenv("RETRIEVAL_EMBEDDING_MODEL_DIR", "models/embeddings")
    RETRIEVAL_SEMANTIC_TIMEOUT_S = _env_float("RETRIEVAL_SEMANTIC_TIMEOUT_S", 1.5)
    # How long warm_up() waits for the model to load before letting it finish
    # in the background (first start: download + torch import).
    RETRIEVAL_MODEL_INIT_TIMEOUT_S = _env_float("RETRIEVAL_MODEL_INIT_TIMEOUT_S", 60)
    RETRIEVAL_QUERY_CACHE_SIZE = int(os.getenv("RETRIEVAL_QUERY_CACHE_SIZE", 2048))

    # --- Sessions --------------------------------------------------------------
    SESSION_TTL_S = _env_float("SESSION_TTL_S", 24 * 3600)
    SESSION_MAX_SESSIONS = int(os.getenv("SESSION_MAX_SESSIONS", 10000))

    # --- API (api/app.py) ------------------------------------------------------
    # Runs orchestrator.warm_up() in the FastAPI lifespan, so the first live
    # request doesn't pay for catalog/prompt/index loading and TLS handshakes.
    WARM_UP_ON_STARTUP = _env_bool("WARM_UP_ON_STARTUP", True)
    # Origins allowed to call /chat from a browser (comma-separated), for the
    # widget served from another host such as the storefront. Empty: same origin only.
    CORS_ALLOW_ORIGINS = tuple(o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if o.strip())


settings = Settings()

if not settings.ZAI_API_KEY:
    logger.error("ZAI_API_KEY is not set (check your .env or deployment secrets).")
