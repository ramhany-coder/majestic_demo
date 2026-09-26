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

    # --- Metadata filter extractor (agents/filter_extractor) -----------------
    # Routes are "router:model" (router is one of llm.helpers.Helpers.routers_list).
    # The primary is the smallest fast model this project can reach; the
    # fallback is tried once per call when the primary errors or times out.
    EXTRACTOR_PRIMARY_ROUTE = os.getenv("EXTRACTOR_PRIMARY_ROUTE", "groq:qwen/qwen3.8-27b")
    EXTRACTOR_FALLBACK_ROUTE = os.getenv("EXTRACTOR_FALLBACK_ROUTE", "groq:openai/gpt-oss-20b")
    # Feature flags for the two fallback stages.
    EXTRACTOR_USE_FALLBACK_MODEL = _env_bool("EXTRACTOR_USE_FALLBACK_MODEL", True)
    EXTRACTOR_USE_RULE_FALLBACK = _env_bool("EXTRACTOR_USE_RULE_FALLBACK", True)
    # Per-attempt timeouts (seconds), and a hard per-call deadline over both attempts.
    EXTRACTOR_TIMEOUT_S = _env_float("EXTRACTOR_TIMEOUT_S", 2.5)
    EXTRACTOR_FALLBACK_TIMEOUT_S = _env_float("EXTRACTOR_FALLBACK_TIMEOUT_S", 2.5)
    EXTRACTOR_CALL_DEADLINE_S = _env_float("EXTRACTOR_CALL_DEADLINE_S", 4.5)
    # Reasoning models (gpt-oss) spend completion tokens thinking before they
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


settings = Settings()

if not settings.GROQ_API:
    logger.error("GROQ_API is not set (check your .env or deployment secrets).")
