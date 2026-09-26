import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# Importing api.endpoints pulls in every agent module, but the catalog, the
# prompt templates, the retrieval index and the cached LLM instances are all
# lazy. orchestrator.warm_up() builds them at startup instead of inside the
# first live request's per-call timeouts, and pre-opens pooled HTTPS
# connections to the primary provider.
from api.chat import router as chat_router
from api.endpoints import router
from agents.orchestrator.orchestrator import warm_up
from config import settings

logger = logging.getLogger("api.app")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs inside the serving event loop, which warm_up needs: the cached model
    # instances bind that loop's shared HTTP client.
    if settings.WARM_UP_ON_STARTUP:
        logger.info("Startup: loading catalog, prompts and retrieval index, building LLM clients...")
        opened = await warm_up()
        logger.info("Warm-up complete (%d pooled connections opened).", opened)
    else:
        logger.info(
            "Skipping warm-up at startup (WARM_UP_ON_STARTUP=False); "
            "everything will load lazily on the first request instead."
        )
    yield


app = FastAPI(title="Majestic Chat Pipeline API", lifespan=lifespan)

app.include_router(router, prefix="/api")
app.include_router(chat_router)            # POST /chat (SSE), GET /api/products/{handle}

# The widget on another origin (the storefront's app embed) needs CORS for /chat.
if settings.CORS_ALLOW_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.CORS_ALLOW_ORIGINS),
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.get("/health")
def health():
    return {"status": "ok"}


# The Jamila widget and its demo page, mounted last so the API routes win.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


# Run from the repo root with:
#   uvicorn api.app:app --reload
# then open http://127.0.0.1:8000/ for the widget.
