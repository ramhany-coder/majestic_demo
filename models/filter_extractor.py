from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class IngredientFilter(BaseModel):
    include: List[str] = Field(default_factory=list)   # ALL must be present
    exclude: List[str] = Field(default_factory=list)   # NONE may be present; never relaxed


class UnmatchedTerms(BaseModel):
    """Terms with no catalog value: soft full-text boost/penalty on description."""
    concerns: List[str] = Field(default_factory=list)
    include: List[str] = Field(default_factory=list)
    exclude: List[str] = Field(default_factory=list)


class FilterMeta(BaseModel):
    latency_ms: float = 0
    # call key -> ok | fallback_model | rule_based | timeout | failed | skipped
    calls: Dict[str, str] = Field(default_factory=dict)
    call_latency_ms: Dict[str, float] = Field(default_factory=dict)
    routes: Dict[str, str] = Field(default_factory=dict)
    prompt_tokens: Dict[str, int] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)
    # Validated output of each call before merging (for tracing and offline re-scoring).
    call_outputs: Dict[str, dict] = Field(default_factory=dict)
    cached: bool = False
    short_circuit: Optional[str] = None
    query_en: Optional[str] = None


class MetadataFilters(BaseModel):
    """Merged output of the 10 extractor calls, applied by the retrieval layer:
    AND across keys, OR within a key (see ARCHITECTURE_NOTES.md)."""
    name_en: List[str] = Field(default_factory=list)
    name_ar: List[str] = Field(default_factory=list)
    matched_handles: List[str] = Field(default_factory=list)
    brand: List[str] = Field(default_factory=list)
    category: List[str] = Field(default_factory=list)
    product_group: List[str] = Field(default_factory=list)
    product_type: List[str] = Field(default_factory=list)
    product_form: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    suitable_for: List[str] = Field(default_factory=list)
    hero_ingredient: List[str] = Field(default_factory=list)
    ingredients: IngredientFilter = Field(default_factory=IngredientFilter)
    unmatched: UnmatchedTerms = Field(default_factory=UnmatchedTerms)
    meta: FilterMeta = Field(default_factory=FilterMeta)

    def is_empty(self) -> bool:
        return not any([
            self.name_en, self.name_ar, self.matched_handles, self.brand, self.category,
            self.product_group, self.product_type, self.product_form, self.concerns,
            self.suitable_for, self.hero_ingredient, self.ingredients.include,
            self.ingredients.exclude, self.unmatched.concerns, self.unmatched.include,
            self.unmatched.exclude,
        ])
