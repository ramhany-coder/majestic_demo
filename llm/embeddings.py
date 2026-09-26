"""Embedding models, built from a "router:model" route like the chat models
in llm_models.py:

    get_embeddings("hf:sentence-transformers/all-MiniLM-L6-v2")   # local, no API key
    get_embeddings("gpt:text-embedding-3-small")
    get_embeddings("gemini:models/text-embedding-004")
    get_embeddings("ollama:nomic-embed-text")

Every router returns a LangChain `Embeddings` object (embed_documents,
embed_query, aembed_query). Provider packages are imported only when their
router is used, so the local model's torch import never runs in tests.

An "hf:" model is downloaded once into RETRIEVAL_EMBEDDING_MODEL_DIR and
loaded from that folder afterwards (model_manager.py), so restarts make no
network call.
"""

from functools import lru_cache
from pathlib import Path

from config import settings

ROOT = Path(__file__).resolve().parents[1]

EMBEDDING_ROUTERS = ("hf", "gpt", "gemini", "ollama")


def parse_embedding_route(route: str) -> tuple:
    router, _, model = (route or "").partition(":")
    router = router.strip().lower()
    if router not in EMBEDDING_ROUTERS:
        raise ValueError(f"Invalid embedding router '{router}'. Must be one of: {list(EMBEDDING_ROUTERS)}")
    if not model.strip():
        raise ValueError(f"Embedding route '{route}' names no model (expected 'router:model').")
    return router, model.strip()


def hf_model_dir(model: str) -> Path:
    from model_manager import model_slug

    base = Path(settings.RETRIEVAL_EMBEDDING_MODEL_DIR)
    return (base if base.is_absolute() else ROOT / base) / model_slug(model)


def _hf(model: str):
    from langchain_huggingface import HuggingFaceEmbeddings

    from model_manager import ensure_hf_model_downloaded

    path = ensure_hf_model_downloaded(model, hf_model_dir(model))
    return HuggingFaceEmbeddings(model_name=str(path), encode_kwargs={"normalize_embeddings": True})


def _gpt(model: str):
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=model, api_key=settings.GPT_API)


def _gemini(model: str):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=settings.GEMINI_API)


def _ollama(model: str):
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=model, base_url=settings.OLLAMA_PATH)


_PROVIDERS = {"hf": _hf, "gpt": _gpt, "gemini": _gemini, "ollama": _ollama}


@lru_cache(maxsize=8)
def get_embeddings(route: str):
    """One shared instance per route (loading a local model takes seconds)."""
    router, model = parse_embedding_route(route)
    return _PROVIDERS[router](model)
