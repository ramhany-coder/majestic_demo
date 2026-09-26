"""Builds/loads the product embedding matrix once per process and reuses it
for every request. Building it means embedding all 108 descriptions (about
2 s on CPU with the default model); loading the cached copy takes a few
milliseconds. Persisted to cache/, keyed by the embedding model and a hash of
data/majestic_catalog.json, and rebuilt automatically the first time either
changes.

Same shape as the drug assistant's agents/meta_data_fiter/engine_registry.py:
a lazy module-level singleton (get_product_embeddings), one cache file at a
time, a schema version in the file name, and a rebuild instead of a crash on
a corrupt file. semantic.py calls it from the embedding model's background
init, so orchestrator.warm_up() pays the cost at startup.
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np

from agents.filter_extractor.catalog import PRODUCTS_PATH, ROOT, _load_json

logger = logging.getLogger("retrieval")

DATA_PATH = PRODUCTS_PATH
CACHE_DIR = ROOT / "cache"
_PREFIX = "product_embeddings"
_WS_RE = re.compile(r"\s+")

# Bump whenever what is embedded (embedding_text) or the saved layout changes.
# The cache key is only the model plus a hash of the *data* file, so without
# this an older cache would load as-is and silently rank with stale vectors.
_SCHEMA_VERSION = 1

_UNSET = object()
_embeddings = _UNSET


@dataclass
class ProductEmbeddings:
    route: str
    handles: List[str]          # row order
    matrix: np.ndarray          # float32, one unit-length row per product


def embedding_text(product: dict) -> str:
    """The English description; for a bundle without one, its name plus the
    names of its components."""
    desc = _WS_RE.sub(" ", product.get("description") or "").strip()
    if desc:
        return desc
    parts = [c.get("name") for c in product.get("bundle_components") or [] if c.get("name")]
    return " | ".join([product["name"]] + parts)


def _source_hash() -> str:
    return hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()[:16]


def _slug(route: str) -> str:
    return re.sub(r"[^a-zA-Z0-9.]+", "-", route).strip("-").lower()


def _cache_path(route: str, source_hash: str) -> Path:
    return CACHE_DIR / f"{_PREFIX}_v{_SCHEMA_VERSION}_{_slug(route)}_{source_hash}.npz"


def _build_and_cache(route: str, embed_documents: Callable[[List[str]], List[List[float]]],
                     cache_path: Path) -> ProductEmbeddings:
    products = _load_json(DATA_PATH)["products"]
    handles = [p["handle"] for p in products]
    matrix = np.asarray(embed_documents([embedding_text(p) for p in products]), dtype=np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Only one cache file should exist at a time -- an old one is dead
        # weight once the catalog or model it was keyed to no longer matches.
        for stale in CACHE_DIR.glob(f"{_PREFIX}_*.npz"):
            stale.unlink(missing_ok=True)
        np.savez(cache_path, matrix=matrix, handles=np.array(handles))
    except OSError as e:  # a read-only disk only costs a re-embed next start
        logger.warning("[retrieval] could not save product embeddings to %s: %s", cache_path, e)
    return ProductEmbeddings(route, handles, matrix)


def get_product_embeddings(route: str, embed_documents: Callable[[List[str]], List[List[float]]]) -> ProductEmbeddings:
    """Return the process-wide product embeddings for `route`, loading or
    building them on the first call. Later calls return the same in-memory
    object immediately; `embed_documents` only runs when there is no usable
    cache file."""
    global _embeddings
    if _embeddings is not _UNSET and _embeddings.route == route:
        return _embeddings

    cache_path = _cache_path(route, _source_hash())
    if cache_path.exists():
        logger.info("Loading cached product embeddings from %s", cache_path)
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                _embeddings = ProductEmbeddings(route, [str(h) for h in data["handles"]], data["matrix"])
            return _embeddings
        except Exception:
            # Belt-and-suspenders alongside _SCHEMA_VERSION: a corrupted or
            # otherwise-incompatible cache file should trigger a rebuild, not
            # take semantic search down.
            logger.exception("Cached product embeddings at %s failed to load -- rebuilding", cache_path)

    logger.info("No usable cache -- embedding %s with %s", DATA_PATH.name, route)
    _embeddings = _build_and_cache(route, embed_documents, cache_path)
    return _embeddings
