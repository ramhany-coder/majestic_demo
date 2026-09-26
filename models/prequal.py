from typing import Any, Dict, Literal

from pydantic import BaseModel, Field

Language = Literal["ar", "arabizi", "en", "mixed"]
Route = Literal["products_only", "needs_response"]
Persona = Literal["customer", "sales_trainee", "doctor", "unknown"]


class PrequalResult(BaseModel):
    """Output of the pre-qualification stage (rewriter + router), see
    ARCHITECTURE_NOTES.md section 9."""
    query_original: str
    query_en: str
    language: Language
    is_follow_up: bool
    route: Route
    needs_retrieval: bool
    intent: str
    persona: Persona
    # True when the rewriter failed on a non-English or follow-up message:
    # query_en is then the raw message, so retrieval skips the metadata
    # filters and uses text search only.
    skip_metadata_filters: bool = False
    # latency_ms: {rewriter, router, total}; status: {rewriter, router} ->
    # ok | fallback_model | default | skipped; plus routes, prompt_tokens,
    # notes and cached.
    meta: Dict[str, Any] = Field(default_factory=dict)
