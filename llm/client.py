from llm.fallback import FallBack
from config import settings

PRIMARY_ROUTER = "anthropic"
PRIMARY_MODEL = "claude-sonnet-5"

SECONDARY_ROUTER = "groq"
SECONDARY_MODEL = "openai/gpt-oss-120b"

TERTIARY_ROUTER = "gpt"
TERTIARY_MODEL = "gpt-4o-mini"

OLLAMA_ROUTER = "ollama"
OLLAMA_MODEL = "llama3.2:1b"

FALLBACK_ORDER = [SECONDARY_ROUTER]  # Groq only for now -- anthropic/gpt creds not configured

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


fallback_kwargs = {
    f"llm_{PRIMARY_ROUTER}": PRIMARY_MODEL,
    f"llm_{SECONDARY_ROUTER}": SECONDARY_MODEL,
    f"llm_{TERTIARY_ROUTER}": TERTIARY_MODEL,
    f"llm_{OLLAMA_ROUTER}": OLLAMA_MODEL,
}

fallback_client = FallBack(**fallback_kwargs)
