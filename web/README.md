# Jamila front ends

Two front ends on the chat pipeline (`agents/orchestrator`):

- **Query console**: type any query and read the reply, plus what the pipeline
  did. Matched products show as the widget's own cards through
  `Jamila.ProductResults` (carousel, Add to cart, Details opening the answer
  card); "Show matched products" (on by default) hides them and leaves the count.
  FastAPI serves it at `/` (`console.html`, `console.js`, reading `POST /chat`,
  whose `done` event carries a `pipeline` summary for the details). In
  Streamlit it is the default mode: native Streamlit elements, with each turn's
  cards in the component's `view: "cards"` iframe, sized to its content.
- **Widget**: the conversational Ask Jamila / اسأل جميلة, a bilingual chat panel
  with product cards. The rest of this page is about the widget.

The widget serves two hosts from one set of files:

| Host | Command | Transport |
|---|---|---|
| FastAPI | `uvicorn api.app:app` → http://127.0.0.1:8000/index.html | `POST /chat` (server-sent events), `GET /api/products/{handle}` |
| Streamlit | `streamlit run streamlit_app.py`, mode "Widget preview" | Streamlit custom-component messages, pipeline in-process |

`web/app.js` detects which host it is in (Streamlit adds `?streamlitUrl=` to
the component iframe) and picks the transport. Both transports deliver the
same events, built in `api/widget.py`.

## Files

| Path | What it is |
|---|---|
| `tokens.json` | Design tokens, the source of truth. Every text colour names the grounds it is legible on. |
| `tokens.css` | Generated from `tokens.json`. Do not edit. |
| `components/bundle.css` | Component styles. Logical properties only, so `dir="rtl"` mirrors everything. |
| `components/bundle.js` | `window.Jamila`: the component factories and `mount()`. Classic script, no framework, no network. |
| `assets/Icons/*.svg` | The icon set, 20px box, 1.5 stroke, `currentColor`. Mirrored into the bundle as `Jamila.Icon(name)`. |
| `app.js` | Boot: the SSE and Streamlit transports, and the demo page's controls. |
| `index.html` | The widget's demo host page, and the Streamlit component page. |
| `console.html`, `console.js` | The query console. |

After editing `tokens.json` or an icon, run `python -m scripts.build_web` (or
`make web`). `--check` fails if the generated files are stale or any token pair
drops under its contrast floor; CI and `tests/test_web.py` run it.

## Deploying on Streamlit Community Cloud

1. Push the repo to GitHub, then create an app at share.streamlit.io.
2. Main file: `streamlit_app.py`. Python: 3.12 (the CPU-only torch pin in
   `requirements.txt` targets Linux and Python 3.12. On another version, pip
   falls back to the default torch wheel, which pulls several GB of CUDA libraries).
3. Secrets: at least `ZAI_API_KEY = "..."`. Any other variable from
   `.env.example` can go there too; `streamlit_app.py` copies secrets into the
   environment before `config.py` reads it.

The first start downloads the embedding model and builds the catalogue index
(about a minute). After that the app keeps one warm pipeline for all
sessions. `.streamlit/config.toml` sets Streamlit's own colours to match the tokens.

## Mounting the bundle elsewhere

```html
<script src="components/bundle.js"></script>
<script>
  const widget = Jamila.mount(document.getElementById('jamila'), {
    mode: 'floating',            // or 'embedded' to fill the host element
    locale: 'ar', dir: 'rtl',    // set separately: locale never controls direction
    audience: 'auto',            // or 'customer' | 'trainee' | 'professional'
    theme: 'light',              // or 'dark' | 'auto'
    transport,                   // { send(request, onEvent), details(handle), reset?(sessionId) }
    cart,                        // optional { add(product) }; default builds a Shopify cart permalink
    onLocaleRequest: (next) => { /* host decides: switch lang, dir, URL */ },
  });
</script>
```

`mount()` builds the widget in a Shadow DOM and links `tokens.css` and
`bundle.css` into it from next to the script. The host page loads the fonts,
because `@font-face` does not apply inside a shadow root.

On the storefront, pass a `cart` whose `add()` posts `/cart/add.js`. The
default cart is for pages outside the store: it builds a
`https://e-majestic.com/cart/{variant}:{qty}` permalink for the toast's
Checkout link. If the widget is served from another origin, list the
storefront in `CORS_ALLOW_ORIGINS`.

## `/chat` contract

`POST /chat` with `{"message": "...", "session_id": "...", "locale": "ar" | "en"}`
returns `text/event-stream`:

| Event | Data | When |
|---|---|---|
| `status` | `{"stage": "understanding" \| "searching" \| "writing"}` | as each pipeline stage starts |
| `message` | `{"text", "path", "language", "health"}` | always; `health: true` adds the disclaimer |
| `products` | `{"items": [card], "total", "relaxed"}` | when the turn shows products |
| `done` | `{"persona", "intent", "route", "path", "timings_ms", "pipeline"}` | last; `pipeline` has the rewritten request, model status and applied filters |
| `error` | `{"code": "pipeline_failed"}` | instead of the above if the pipeline raises |

A card carries `Jamila.CARD_FIELDS` / `api.widget.CARD_FIELDS` only (a test
keeps the two lists equal). `description`, `key_ingredients`, `how_to_use` and
`warnings` come from `GET /api/products/{handle}` for the answer card; `tags`,
`collections` and `sku` are never sent.

## Content rules the code enforces

- **Disclaimer.** A reply is flagged `health` when it comes from the responder for a
  health intent, or when the router fell back to its default intent. Every answer card
  carries the disclaimer too.
- **No machine translation of health copy.** The answer card reads `how_to_use_ar`,
  `warnings_ar` and `description_ar` when the catalogue has them. Otherwise it shows the
  English field with `lang="en"` and a note saying so. Today the catalogue has Arabic
  descriptions but English-only how-to-use steps and warnings.
- **"Why this product" line.** Built from the requested filter values the product
  actually has, else its own first concerns. Arabic labels come from a fixed table
  in the bundle (`Jamila.vocab`), and a test fails if a catalogue value is missing
  from it. The line never claims a product is free of an ingredient, because
  `key_ingredients` is not a full ingredient list.
- **Voice.** No emoji and no exclamation marks, in the widget's own copy and in
  the server's templated replies (tested). The customer greeting is the brief's
  approved line, "قوليلي بشرتك محتاجة إيه وأنا ألاقي لك الروتين المناسب.", which
  uses the feminine form. Every other string is gender-neutral.

## Replace first when the theme files land

These are marked `inferred` in `tokens.json`. Each is a one-line change there,
followed by `make web`:

1. `color.brand` (#f46e34) and `color.deep` (#670010): from a brand registry, not the theme CSS.
2. `radius`: read off product-grid screenshots.
3. `font`: IBM Plex Sans / IBM Plex Sans Arabic are the brief's fallback.
4. The avatar: `assets/Icons/mark.svg` is a placeholder line mark. There is no
   Majestic logo in this system, and it should not be redrawn from memory.
