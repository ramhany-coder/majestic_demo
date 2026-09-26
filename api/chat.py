"""The Jamila widget's routes: POST /chat (server-sent events) and
GET /api/products/{handle} (answer card details).

POST /chat {"message": "...", "session_id": "...", "locale": "ar" | "en", "llm": "glm" | "groq" (optional)}
streams each event as the pipeline produces it:

    event: status    data: {"stage": "understanding" | "searching" | "writing"}   (0..3 of these)
  product list or small talk:
    event: message   data: {"text", "path", "language", "health"}
    event: products  data: {"items": [card, ...], "total", "relaxed"}              (only when cards are shown)
  responder:
    event: products  data: {...}                                                   (when retrieval found any)
    event: card      data: {"type": "how_to_use" | "safety" | "compare", ...}      (0..2, built in code)
    event: message   data: {"delta", "path", "language"}                           (the answer as it streams)
    event: message   data: {"text", "replace": true, ...}                          (a guardrail changed the text)
    event: card      data: {"type": "quiz" | "objection", ...}                     (sales training only)
    event: message   data: {"delta", "disclaimer": true, ...}                      (when the disclaimer applies)
  always last:
    event: done      data: {"persona", "intent", "route", "path", "timings_ms", "latency_ms", "partial", "pipeline"}

or, if the pipeline fails, `event: error data: {"code": "pipeline_failed"}`.
Cards and the other payloads are built in api/widget.py.
"""

import asyncio
import json
import logging
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents.orchestrator.orchestrator import handle_message
from api.widget import done_event, error_events, product_details, wire_event

router = APIRouter()
logger = logging.getLogger("api.chat")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", min_length=1, max_length=128)
    locale: Literal["ar", "en"] = "en"      # the widget's copy language; replies follow the message
    llm: Optional[Literal["glm", "groq"]] = None   # hidden provider switch; None: the configured routes


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(payload: ChatRequest):
    queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()

    def on_event(name: str, data: dict) -> None:
        ev = wire_event(name, data)
        queue.put_nowait(sse(ev["event"], ev["data"]))

    async def run() -> None:
        try:
            turn = await handle_message(payload.message, payload.session_id, llm=payload.llm,
                                        on_stage=lambda stage: queue.put_nowait(sse("status", {"stage": stage})),
                                        on_event=on_event)
            events = [done_event(turn)]
        except Exception:  # noqa: BLE001 -- the stream must still end with an event
            logger.exception("[chat] pipeline failed for session=%s", payload.session_id)
            events = error_events()
        for ev in events:
            queue.put_nowait(sse(ev["event"], ev["data"]))
        queue.put_nowait(None)

    async def stream():
        task = asyncio.create_task(run())
        try:
            while (frame := await queue.get()) is not None:
                yield frame
        finally:
            if not task.done():          # the client went away mid-turn
                task.cancel()

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/api/products/{handle}")
def product(handle: str):
    details = product_details(handle)
    if details is None:
        raise HTTPException(status_code=404, detail=f"No product with handle '{handle}'")
    return details
