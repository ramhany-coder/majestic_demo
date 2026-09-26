from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Conflict = Literal["contains_excluded", "filter_mismatch"]
Fallback = Literal["semantic_only", "no_semantic"]


class RetrievedProduct(BaseModel):
    """One product card. `score` is the name score for a name hit, else the
    fusion score (see ARCHITECTURE_NOTES.md section 10)."""
    handle: str
    name: str
    name_ar: Optional[str] = None
    brand: str
    product_type: str
    product_form: Optional[str] = None
    price: float
    compare_at_price: Optional[float] = None
    promotion: Optional[str] = None
    available: bool = True
    url: str
    url_ar: Optional[str] = None
    image: Optional[str] = None             # first image src
    score: float = 0.0
    # name_score, sem_score, filters_matched, boosts, conflict (None | Conflict)
    match: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Output of the retrieval agent: name hits first, then the filtered
    candidates ranked by fusion score, cut to k."""
    products: List[RetrievedProduct] = Field(default_factory=list)   # length <= k
    k: int
    total_candidates: int = 0                   # |C| before the cut
    applied_filters: Dict[str, List[str]] = Field(default_factory=dict)   # the keys actually used
    relaxed_keys: List[str] = Field(default_factory=list)
    name_hits: List[str] = Field(default_factory=list)                   # handles matched by name
    unresolved_names: List[str] = Field(default_factory=list)
    fallback: Optional[Fallback] = None
    # latency_ms: {filters, names, semantic, fusion, total}; semantic status;
    # relaxed_values; bundles_allowed; notes.
    meta: Dict[str, Any] = Field(default_factory=dict)

    @property
    def handles(self) -> List[str]:
        return [p.handle for p in self.products]
