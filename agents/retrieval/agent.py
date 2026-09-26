"""Retrieval agent: extractor filters + semantic scores -> the top k products.
Deterministic and in-memory: no LLM, a few milliseconds.

    filters, sem = await asyncio.gather(extract_filters(query_en), semantic_search(query_en))
    result = retrieve(filters, sem, query_en, intent=pq.intent, k=pq.k)

Order of the result (ARCHITECTURE_NOTES.md section 10):
1. Name hits (name_search.py), best first, whether or not they pass the
   filters: the user named them. A hit containing an excluded ingredient is
   kept with conflict "contains_excluded"; one failing another requested
   filter gets "filter_mismatch".
2. Then, for the RETRIEVAL_FILL_INTENTS (find / refine) or when no name was
   resolved, the filtered candidates C (filters.py) ranked by fusion.py. By
   default C holds only products matching every requested key, so it can be
   empty; meta.near_miss_keys then names the keys that stood in the way.
Duplicates are removed and the list is cut to k.
"""

import logging
import time
from typing import List, Optional

from agents.retrieval.filters import apply_filters
from agents.retrieval.fusion import rank
from agents.retrieval.index_builder import RetrievalIndex, get_index
from agents.retrieval.name_search import NameSearchResult, search_names
from agents.retrieval.semantic import SemanticScores
from config import settings
from models.filter_extractor import MetadataFilters
from models.retrieval import RetrievalResult, RetrievedProduct

logger = logging.getLogger("retrieval")

FALLBACK_NO_SEMANTIC = "no_semantic"


def resolve_k(k: Optional[int]) -> int:
    """The router's k, or RETRIEVAL_K_DEFAULT when missing; capped at RETRIEVAL_K_MAX."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        k = settings.RETRIEVAL_K_DEFAULT
    return max(1, min(k, settings.RETRIEVAL_K_MAX))


def _card(index: RetrievalIndex, handle: str, score: float, match: dict) -> RetrievedProduct:
    p = index.products[handle]
    images = p.get("images") or []
    return RetrievedProduct(
        handle=handle,
        name=p["name"],
        name_ar=p.get("name_ar"),
        brand=p["brand"],
        product_type=p["product_type"],
        product_form=p.get("product_form"),
        price=p["price"],
        compare_at_price=p.get("compare_at_price"),
        promotion=p.get("promotion"),
        available=p.get("available", True),
        url=p["url"],
        url_ar=p.get("url_ar"),
        image=images[0].get("src") if images else None,
        score=round(score, 4),
        match=match,
    )


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 2)


def retrieve(filters: Optional[MetadataFilters], sem: Optional[SemanticScores] = None, query_en: str = "",
             intent: Optional[str] = None, k: Optional[int] = None,
             index: Optional[RetrievalIndex] = None) -> RetrievalResult:
    """`filters` is None when the extractor was skipped (the rewriter failed);
    `sem` is None when the semantic search did not run. `query_en` is only
    logged: the semantic search already used it."""
    start = time.perf_counter()
    index = index or get_index()
    k = resolve_k(k)
    sem = sem or SemanticScores()

    t = time.perf_counter()
    names = NameSearchResult()
    if filters is not None and (filters.name_en or filters.name_ar):
        names = search_names(filters.name_en, filters.name_ar, index.names_en, index.names_ar)
    names_ms = _ms(t)

    t = time.perf_counter()
    named_bundle = any(h.handle in index.bundles for h in names.hits)
    fo = apply_filters(index, filters, intent, named_bundle)
    filters_ms = _ms(t)

    t = time.perf_counter()
    exclude = set(filters.ingredients.exclude) if filters is not None else set()
    products: List[RetrievedProduct] = []
    for hit in names.hits:
        conflict = None
        if index.ingredients[hit.handle] & exclude:
            conflict = "contains_excluded"
        elif fo.mismatched_keys(hit.handle):
            conflict = "filter_mismatch"
        products.append(_card(index, hit.handle, hit.score, {
            "source": "name", "name_score": round(hit.score, 4), "name_dice": round(hit.dice, 4),
            "sem_score": round(sem.scores.get(hit.handle, 0.0), 4),
            "filters_matched": fo.matched_keys(hit.handle), "boosts": [], "conflict": conflict,
        }))
    products = products[:k]

    if not names.hits or intent in settings.RETRIEVAL_FILL_INTENTS:
        seen = {p.handle for p in products}
        for r in rank(index, fo.candidates, filters, sem.scores, names.soft):
            if len(products) >= k:
                break
            if r.handle in seen:
                continue
            products.append(_card(index, r.handle, r.score, {
                "source": "filters", "name_score": round(names.soft.get(r.handle, 0.0), 4),
                "sem_score": round(r.sem, 4), "filters_matched": fo.matched_keys(r.handle),
                "boosts": r.boosts, "conflict": None,
            }))
    fusion_ms = _ms(t)

    fallbacks = ([] if sem.ok else [FALLBACK_NO_SEMANTIC]) + ([fo.fallback] if fo.fallback else [])
    result = RetrievalResult(
        products=products,
        k=k,
        total_candidates=len(fo.candidates),
        applied_filters=fo.applied,
        relaxed_keys=fo.relaxed_keys,
        name_hits=names.handles,
        unresolved_names=names.unresolved,
        fallback=fallbacks[0] if fallbacks else None,
        meta={
            "latency_ms": {"names": names_ms, "filters": filters_ms, "semantic": sem.latency_ms,
                           "fusion": fusion_ms, "total": _ms(start)},
            "semantic": sem.status,
            "fallbacks": fallbacks,
            "relaxed_values": {key: fo.requested[key] for key in fo.relaxed_keys},
            "near_miss_keys": fo.near_miss_keys,
            "implied_keys": fo.implied_keys,
            "bundles_allowed": fo.bundles_allowed,
            "excluded_count": len(fo.excluded),
            "filters_skipped": filters is None,
        },
    )
    if not sem.ok:
        logger.warning("[retrieval] semantic: %s (%s); ranking on boosts and tie-breakers", sem.status, sem.error)
    logger.info("[retrieval] %.1fms intent=%s k=%d candidates=%d relaxed=%s names=%s unresolved=%s fallback=%s "
                "query_en=%r top=%s", result.meta["latency_ms"]["total"], intent, k, result.total_candidates,
                result.relaxed_keys, result.name_hits, result.unresolved_names, result.fallback, query_en,
                result.handles[:3])
    return result
