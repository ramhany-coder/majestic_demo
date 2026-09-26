"""Jamila on Streamlit: the same widget as web/, mounted as a custom
component, with the chat pipeline running in this process.

    streamlit run streamlit_app.py

The widget sends each request (a chat message, an answer card's details, a
reset) as the component value. This script handles it and passes the result
back in the component's args, as the same events the /chat SSE stream sends
(api/widget.py), so web/app.js renders both transports with one code path.

Deploys to Streamlit Community Cloud as it is: main file streamlit_app.py,
Python 3.12, and GROQ_API (plus any other setting from .env.example) in the
app's secrets.
"""

import asyncio
import logging
import os
import threading
import uuid
from pathlib import Path

import streamlit as st


def _export_secrets() -> None:
    """Streamlit secrets -> environment variables, before config.py reads them."""
    try:
        items = dict(st.secrets)
    except Exception:  # noqa: BLE001 -- no secrets file: a local run reads .env instead
        return
    for key, value in items.items():
        if isinstance(value, (str, int, float, bool)):
            os.environ.setdefault(key, str(value))


_export_secrets()

import streamlit.components.v1 as components  # noqa: E402

from agents.orchestrator.orchestrator import handle_message, warm_up  # noqa: E402
from api.widget import error_events, product_details, turn_events  # noqa: E402
from config import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("streamlit_app")

WEB_DIR = Path(__file__).resolve().parent / "web"
FRAME_HEIGHT = int(os.getenv("JAMILA_FRAME_HEIGHT", "720"))
TURN_TIMEOUT_S = 120
MAX_DETAILS = 6          # answer-card payloads kept in the component args
AUDIENCES = {
    "auto": "Detected from the chat",
    "customer": "Customer",
    "trainee": "Sales trainee",
    "professional": "Doctor or pharmacist",
}

st.set_page_config(page_title="Ask Jamila", layout="centered")
st.markdown(
    """<style>
    .block-container { padding-top: 3.5rem; padding-bottom: 1rem; }   /* clears the toolbar */
    /* Keep the widget at full opacity while a turn runs. */
    [data-stale="true"], .stale-element { opacity: 1 !important; }
    </style>""",
    unsafe_allow_html=True,
)

jamila = components.declare_component("jamila", path=str(WEB_DIR))


@st.cache_resource(show_spinner=False)
def pipeline_loop() -> asyncio.AbstractEventLoop:
    """One event loop for the whole process, on its own thread. warm_up() binds
    the cached LLM clients to the loop it runs in, so every turn runs on this
    same loop; asyncio.run() per rerun would give each turn a new one."""
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, name="jamila-pipeline", daemon=True).start()
    try:
        opened = asyncio.run_coroutine_threadsafe(warm_up(), loop).result(timeout=600)
        logger.info("Warm-up complete (%d pooled connections opened).", opened)
    except Exception:  # noqa: BLE001 -- everything also loads lazily on the first message
        logger.exception("Warm-up failed; the pipeline will load on the first message instead.")
    return loop


def run_turn(message: str, session_id: str):
    future = asyncio.run_coroutine_threadsafe(handle_message(message, session_id), pipeline_loop())
    return future.result(timeout=TURN_TIMEOUT_S)


def init_state() -> None:
    defaults = {
        "jamila_session": str(uuid.uuid4()),
        "jamila_turns": [],          # [{id, message, events}]
        "jamila_details": {},        # request id -> details (None: not found)
        "jamila_reset": None,        # id of the last reset request
        "jamila_handled": set(),     # request ids already handled
        "jamila_generation": 0,      # bumped to remount the component
        "jamila_last_turn": None,    # the last TurnResult, for the sidebar
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def new_conversation(session_id: str = "") -> None:
    st.session_state.jamila_session = session_id or str(uuid.uuid4())
    st.session_state.jamila_turns = []
    st.session_state.jamila_details = {}
    st.session_state.jamila_last_turn = None


def handle(request: dict) -> None:
    """One request from the widget. Results go into session state before any
    other Streamlit call, so a rerun requested meanwhile cannot drop them."""
    state = st.session_state
    rid = request.get("id")
    kind = request.get("type")
    if kind == "chat":
        message = str(request.get("message") or "")
        session = str(request.get("session_id") or state.jamila_session)
        state.jamila_session = session
        try:
            turn = run_turn(message, session)
            events = turn_events(turn)
            state.jamila_last_turn = turn.model_dump(exclude={"products"})
        except Exception:  # noqa: BLE001 -- the widget still gets an answer
            logger.exception("Turn failed for session=%s", session)
            events = error_events()
        state.jamila_turns = state.jamila_turns + [{"id": rid, "message": message, "events": events}]
    elif kind == "details":
        details = dict(state.jamila_details)
        details[rid] = product_details(str(request.get("handle") or ""))
        state.jamila_details = dict(list(details.items())[-MAX_DETAILS:])
    elif kind == "reset":
        new_conversation(str(request.get("session_id") or ""))
        state.jamila_reset = rid


init_state()

with st.sidebar:
    st.subheader("Jamila")
    audience = st.selectbox("Audience", list(AUDIENCES), format_func=AUDIENCES.get,
                            help="Sets the greeting, the suggested questions and the card density. "
                                 "'Detected from the chat' follows the persona the router detects.")
    if st.button("New conversation", width="stretch"):
        new_conversation()
        st.session_state.jamila_generation += 1
    if not settings.GROQ_API:
        st.warning("GROQ_API is not set. Add it to the app's secrets (or to .env locally); "
                   "until then replies fall back to defaults.")
    last = st.session_state.jamila_last_turn
    if last:
        with st.expander("Last turn in the pipeline"):
            pq = last["prequal"]
            st.markdown(f"**Path** `{last['path']}` · **intent** `{pq['intent']}` · "
                        f"**persona** `{pq['persona']}` · **language** `{pq['language']}`")
            st.markdown(f"**Rewritten request** {pq['query_en']}")
            if last.get("retrieval"):
                st.markdown("**Applied filters**")
                st.json(last["retrieval"].get("applied_filters") or {}, expanded=False)
            st.markdown("**Timings (ms)**")
            st.json(last.get("timings_ms") or {}, expanded=False)

with st.spinner("Loading the catalogue and models"):
    pipeline_loop()

value = jamila(
    turns=st.session_state.jamila_turns,
    details=st.session_state.jamila_details,
    reset=st.session_state.jamila_reset,
    session_id=st.session_state.jamila_session,
    audience=audience,
    height=FRAME_HEIGHT,
    key=f"jamila-{st.session_state.jamila_generation}",
    default=None,
)

if isinstance(value, dict) and value.get("id") and value["id"] not in st.session_state.jamila_handled:
    st.session_state.jamila_handled.add(value["id"])
    handle(value)
    st.rerun()
