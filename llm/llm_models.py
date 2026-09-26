import asyncio
import weakref
from functools import lru_cache

import httpx

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from llm.helpers import Helpers
from config import settings


class Llm:
    routers_list = ["anthropic", "gemini", "gpt", "groq", "ollama", "zai"]

    def __init__(self, temp: float = 0):
        self.temp = temp

    def get_model(self, router: str, model: str, **overrides):
        """Build a chat model. `overrides` go straight to the LangChain
        constructor (max_tokens, max_retries, reasoning_effort, temperature...)."""
        router = Helpers.validate_router(router)

        providers = {
            "anthropic": Llm.anthropic,
            "gemini": Llm.gemini,
            "groq": Llm.groq,
            "ollama": Llm.ollama,
            "gpt": Llm.gpt,
            "zai": Llm.zai,
        }

        temp = overrides.pop("temperature", self.temp)
        return providers[router](model, temp, **overrides)

    def get_cached_model(self, router: str, model: str, **overrides):
        """Same as get_model, but returns one shared instance per distinct
        (router, model, overrides). Groq / OpenAI / Z.ai instances created inside an
        event loop also share that loop's single httpx.AsyncClient, so all
        concurrent calls draw from one connection pool."""
        router = Helpers.validate_router(router)
        if router in _SHARED_HTTP_ROUTERS and "http_async_client" not in overrides:
            client = shared_async_http_client()
            if client is not None:
                overrides["http_async_client"] = client
        return _cached_model(self, router, model, tuple(sorted(overrides.items(), key=lambda kv: kv[0])))

    # 0. Anthropic Claude
    @staticmethod
    def anthropic(model: str, temp: float, **kw):
        kw.pop("reasoning_effort", None)
        return ChatAnthropic(
            model=model,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=temp,
            **kw,
        )

    # 1. Google Gemini
    @staticmethod
    def gemini(model: str, temp: float, **kw):
        kw.pop("reasoning_effort", None)
        if "max_tokens" in kw:
            kw["max_output_tokens"] = kw.pop("max_tokens")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.GEMINI_API,
            temperature=temp,
            **kw,
        )

    # 2. Groq
    @staticmethod
    def groq(model: str, temp: float, **kw):
        return ChatGroq(
            model=model,
            api_key=settings.GROQ_API,
            temperature=temp,
            **kw,
        )

    # 3. Local Ollama
    @staticmethod
    def ollama(model: str, temp: float, **kw):
        if "max_tokens" in kw:
            kw["num_predict"] = kw.pop("max_tokens")
        kw.pop("max_retries", None)
        kw.pop("reasoning_effort", None)
        return ChatOllama(
            model=model,
            base_url=settings.OLLAMA_PATH,
            temperature=temp,
            num_ctx=2048,
            keep_alive="0s",
            **kw,
        )

    # 4. OpenAI GPT
    @staticmethod
    def gpt(model: str, temp: float, **kw):
        return ChatOpenAI(
            model=model,
            api_key=settings.GPT_API,
            temperature=temp,
            **kw,
        )

    # 5. Z.ai GLM, over its OpenAI-compatible endpoint
    @staticmethod
    def zai(model: str, temp: float, **kw):
        return ChatOpenAI(
            model=model,
            api_key=settings.ZAI_API_KEY,
            base_url=settings.ZAI_BASE_URL,
            temperature=temp,
            **kw,
        )


_SHARED_HTTP_ROUTERS = {"groq", "gpt", "zai"}
_loop_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]" = weakref.WeakKeyDictionary()
HTTP_POOL_LIMITS = httpx.Limits(max_connections=64, max_keepalive_connections=32, keepalive_expiry=120)


def shared_async_http_client():
    """One httpx.AsyncClient per running event loop (None outside a loop).
    httpx clients must not be shared across loops, hence the per-loop map."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    client = _loop_clients.get(loop)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(limits=HTTP_POOL_LIMITS, timeout=httpx.Timeout(30.0, connect=5.0))
        _loop_clients[loop] = client
    return client


@lru_cache(maxsize=256)
def _cached_model(llm: "Llm", router: str, model: str, overrides: tuple):
    return llm.get_model(router, model, **dict(overrides))


client_llm = Llm(temp=0.1)
