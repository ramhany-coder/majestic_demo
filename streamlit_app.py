"""Jamila on Streamlit, with the chat pipeline running in this process.

    streamlit run streamlit_app.py

Two modes, picked in the sidebar:

- Query console (the default): type any query and read the reply, plus what
  the pipeline did. The matched products show as the widget's product cards
  (the same component, in its 'cards' view); "Show matched products" off
  leaves just their count.
- Widget preview: the storefront widget from web/, with product cards, as a
  custom component. The widget sends each request (a chat message, an answer
  card's details, a reset) as the component value; this script handles it and
  passes the result back in the component's args, as the same events the
  /chat SSE stream sends (api/widget.py).

Deploys to Streamlit Community Cloud as it is: main file streamlit_app.py,
Python 3.12, and ZAI_API_KEY (plus any other setting from .env.example) in the
app's secrets.
"""

import asyncio
import html
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
from api.widget import cards, error_events, product_details, turn_events  # noqa: E402
from config import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("streamlit_app")

WEB_DIR = Path(__file__).resolve().parent / "web"
FRAME_HEIGHT = int(os.getenv("JAMILA_FRAME_HEIGHT", "720"))
TURN_TIMEOUT_S = 120
MAX_DETAILS = 6          # answer-card payloads kept in the component args
CONSOLE, WIDGET = "Query console", "Widget preview"
QUERY_EXAMPLE = "Type any query, for example: عايزة سيروم للشعر من غير سيليكون"
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
        "console_session": str(uuid.uuid4()),
        "console_turns": [],         # [{query, turn}] or [{query, error}]
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ---------------------------------------------------------------- query console

def show_text(text: str) -> None:
    """Message text, laid out right to left when it is Arabic. text-align:
    start, because Streamlit sets left explicitly."""
    body = html.escape(text or "").replace("\n", "<br>")
    st.markdown(f'<div dir="auto" style="text-align: start">{body}</div>', unsafe_allow_html=True)


def console_turn(query: str) -> dict:
    try:
        turn = run_turn(query, st.session_state.console_session)
        return {"query": query, "turn": turn.model_dump(mode="json")}
    except Exception as e:  # noqa: BLE001 -- shown in the console instead
        logger.exception("Console turn failed")
        return {"query": query, "error": f"{type(e).__name__}: {e}"}


def show_cards(t: dict, key: str) -> None:
    """The turn's products as the widget's cards (the component's 'cards'
    view), with each card's answer-card details sent along."""
    retrieval = t.get("retrieval") or {}
    items = cards(t["products"], retrieval.get("applied_filters") or {})
    jamila(
        view="cards",
        products=items,
        details={c["handle"]: product_details(c["handle"]) for c in items},
        locale="ar" if t["prequal"]["language"] in ("ar", "mixed") else "en",
        key=key,
        default=None,
    )


def show_details(t: dict) -> None:
    pq = t["prequal"]
    retrieval = t.get("retrieval")
    status = (pq.get("meta") or {}).get("status") or {}
    route = [pq["route"], "retrieval" if pq["needs_retrieval"] else "no retrieval"]
    if pq.get("is_follow_up"):
        route.append("follow-up")
    lines = [
        f"**Rewritten request:** {html.escape(pq['query_en'])}",
        f"**Route:** {' · '.join(route)}",
        "**Model status:** " + (" · ".join(f"{k} {v}" for k, v in status.items()) or "n/a"),
    ]
    if retrieval:
        lines.append(f"**Candidates:** {retrieval.get('total_candidates')}")
        if retrieval.get("relaxed_keys"):
            lines.append(f"**Relaxed filters:** {', '.join(retrieval['relaxed_keys'])}")
    st.markdown("  \n".join(lines))
    if retrieval:
        st.markdown("**Applied filters**")
        st.json({k: v for k, v in (retrieval.get("applied_filters") or {}).items() if v}, expanded=True)
    else:
        st.markdown("Retrieval did not run for this query.")
    st.markdown("**Timings (ms)**")
    st.json(t.get("timings_ms") or {}, expanded=False)


def show_console_turn(entry: dict, index: int, products_on: bool, details_on: bool) -> None:
    with st.chat_message("user"):
        show_text(entry["query"])
    with st.chat_message("assistant"):
        if "error" in entry:
            st.error(f"The pipeline failed: {entry['error']}")
            return
        t = entry["turn"]
        pq = t["prequal"]
        show_text(t["reply"])
        seconds = (t.get("timings_ms") or {}).get("total", 0) / 1000
        st.caption(" · ".join([t["path"], pq["intent"], pq["persona"], pq["language"], f"{seconds:.2f} s"]))
        products = t.get("products") or []
        if products:
            count = "1 product matched" if len(products) == 1 else f"{len(products)} products matched"
            if products_on:
                show_cards(t, key=f"cards-{st.session_state.console_session}-{index}")   # carries its own count
            else:
                st.caption(f"{count}. Turn on 'Show matched products' in the sidebar to see them.")
        if details_on:
            with st.expander("Pipeline details"):
                show_details(t)


def render_console(products_on: bool, details_on: bool) -> None:
    st.subheader("Jamila console")
    st.caption("Any query, in Arabic, Arabizi or English, through the full pipeline: "
               "pre-qualification, filter extraction, retrieval and the reply. "
               "Follow-ups use the conversation so far, until you start a new session.")
    for index, entry in enumerate(st.session_state.console_turns):
        show_console_turn(entry, index, products_on, details_on)
    query = (st.chat_input(QUERY_EXAMPLE) or "").strip()
    if query:
        with st.chat_message("user"):
            show_text(query)
        with st.chat_message("assistant"), st.spinner("Running the pipeline"):
            # Stored before the next Streamlit call, so a rerun cannot drop it.
            st.session_state.console_turns = st.session_state.console_turns + [console_turn(query)]
        st.rerun()


# ---------------------------------------------------------------- widget preview

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


def widget_sidebar() -> str:
    audience = st.selectbox("Audience", list(AUDIENCES), format_func=AUDIENCES.get,
                            help="Sets the greeting, the suggested questions and the card density. "
                                 "'Detected from the chat' follows the persona the router detects.")
    if st.button("New conversation", width="stretch"):
        new_conversation()
        st.session_state.jamila_generation += 1
    last = st.session_state.jamila_last_turn
    if last:
        with st.expander("Last turn in the pipeline"):
            pq = last["prequal"]
            st.markdown(f"**Path** `{last['path']}` · **intent** `{pq['intent']}` · "
                        f"**persona** `{pq['persona']}` · **language** `{pq['language']}`")
            st.markdown(f"**Rewritten request** {html.escape(pq['query_en'])}")
            if last.get("retrieval"):
                st.markdown("**Applied filters**")
                st.json(last["retrieval"].get("applied_filters") or {}, expanded=False)
            st.markdown("**Timings (ms)**")
            st.json(last.get("timings_ms") or {}, expanded=False)
    return audience


def render_widget(audience: str) -> None:
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


# ---------------------------------------------------------------- page

init_state()

with st.sidebar:
    st.subheader("Jamila")
    mode = st.radio("Mode", [CONSOLE, WIDGET],
                    help="Query console: any query, with the reply and the pipeline's details. "
                         "Widget preview: the storefront widget, with product cards.")
    if mode == CONSOLE:
        products_on = st.toggle("Show matched products", value=True,
                                help="Show the products retrieval matched under each reply, as the "
                                     "widget's product cards. Off: only how many matched.")
        details_on = st.toggle("Show pipeline details", value=True)
        if st.button("New session", width="stretch"):
            st.session_state.console_session = str(uuid.uuid4())
            st.session_state.console_turns = []
    else:
        audience = widget_sidebar()
    if not settings.ZAI_API_KEY:
        st.warning("ZAI_API_KEY is not set. Add it to the app's secrets (or to .env locally); "
                   "until then replies fall back to defaults.")

with st.spinner("Loading the catalogue and models"):
    pipeline_loop()

if mode == CONSOLE:
    render_console(products_on, details_on)
else:
    render_widget(audience)
