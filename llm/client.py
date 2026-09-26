from contextlib import contextmanager
from contextvars import ContextVar
from typing import List, Optional

from llm.fallback import FallBack
from config import settings

PRIMARY_ROUTER = "anthropic"
PRIMARY_MODEL = "claude-sonnet-5"

SECONDARY_ROUTER = "groq"
SECONDARY_MODEL = "openai/gpt-oss-120b"

ZAI_ROUTER = "zai"
ZAI_MODEL = "glm-5.3-flash"

TERTIARY_ROUTER = "gpt"
TERTIARY_MODEL = "gpt-4o-mini"

OLLAMA_ROUTER = "ollama"
OLLAMA_MODEL = "llama3.2:1b"

FALLBACK_ORDER = [ZAI_ROUTER]  # every LLM call goes to Z.ai GLM

# Metadata filter extractor: its own short chain of small fast models, written
# as "router:model" routes (see llm.fallback.parse_route). The fallback route is
# dropped when EXTRACTOR_USE_FALLBACK_MODEL is off.
EXTRACTOR_FALLBACK_ORDER = [settings.EXTRACTOR_PRIMARY_ROUTE] + (
    [settings.EXTRACTOR_FALLBACK_ROUTE]
    if settings.EXTRACTOR_USE_FALLBACK_MODEL and settings.EXTRACTOR_FALLBACK_ROUTE
    else []
)

# Pre-qualification stage (rewriter + router): same shape, its own routes.
PREQUAL_FALLBACK_ORDER = [settings.PREQUAL_PRIMARY_ROUTE] + (
    [settings.PREQUAL_FALLBACK_ROUTE]
    if settings.PREQUAL_USE_FALLBACK_MODEL and settings.PREQUAL_FALLBACK_ROUTE
    else []
)

# Responder (streamed answer + sales card): quality tier first, then the fast tier.
RESPONDER_FALLBACK_ORDER = [settings.RESPONDER_PRIMARY_ROUTE] + (
    [settings.RESPONDER_FALLBACK_ROUTE]
    if settings.RESPONDER_USE_FALLBACK_MODEL and settings.RESPONDER_FALLBACK_ROUTE
    else []
)


fallback_kwargs = {
    f"llm_{PRIMARY_ROUTER}": PRIMARY_MODEL,
    f"llm_{SECONDARY_ROUTER}": SECONDARY_MODEL,
    f"llm_{TERTIARY_ROUTER}": TERTIARY_MODEL,
    f"llm_{OLLAMA_ROUTER}": OLLAMA_MODEL,
    f"llm_{ZAI_ROUTER}": ZAI_MODEL,
}

fallback_client = FallBack(**fallback_kwargs)


# ---------------------------------------------------------------- per-turn provider
# handle_message(..., llm="groq") runs the turn inside use_provider("groq").
# Tasks the turn starts copy the context, so every call they make sees it too.
LLM_PROVIDERS = tuple(settings.LLM_PROVIDER_ROUTES)
_turn_provider: ContextVar[Optional[str]] = ContextVar("llm_provider", default=None)


def current_provider() -> Optional[str]:
    return _turn_provider.get()


@contextmanager
def use_provider(provider: Optional[str]):
    """Route this turn's LLM calls to `provider` (None or unknown: the configured routes)."""
    token = _turn_provider.set(provider if provider in settings.LLM_PROVIDER_ROUTES else None)
    try:
        yield
    finally:
        _turn_provider.reset(token)


def turn_routes(default: List[str]) -> List[str]:
    """The chosen provider's routes for this turn, else `default`."""
    provider = _turn_provider.get()
    if provider is None:
        return list(default)
    routes = [r for r in settings.LLM_PROVIDER_ROUTES[provider] if r]
    use_fallback = len(default) > 1          # keeps the *_USE_FALLBACK_MODEL switch
    return routes if use_fallback else routes[:1]
