/* Boot for web/index.html: picks a transport and mounts the widget.

   - Served by FastAPI (api/app.py): POST /chat streams server-sent events and
     GET /api/products/{handle} returns an answer card's details.
   - Inside Streamlit (streamlit_app.py): this page is a custom component's
     iframe (Streamlit adds ?streamlitUrl=...). Requests go to Python as the
     component value; replies come back in the render args, as the same events.

   The host page decides what a language switch means: here it sets the
   document's lang and dir, then the widget's locale and direction. */
(function () {
  'use strict';

  const inStreamlit = new URLSearchParams(location.search).has('streamlitUrl');
  const host = document.getElementById('jamila');

  function newId() {
    return (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : 'r-' + Date.now().toString(36) + Math.random().toString(36).slice(2);
  }

  function dirFor(locale) {
    return locale === 'ar' ? 'rtl' : 'ltr';
  }

  function readPref(key) {
    try { return localStorage.getItem('jamila.' + key); } catch (e) { return null; }
  }

  function writePref(key, value) {
    try { localStorage.setItem('jamila.' + key, value); } catch (e) { /* private mode */ }
  }

  // ---------------------------------------------------------------- hidden LLM switch

  // No UI: open the page with ?llm=groq or ?llm=glm and every later request
  // from this browser asks /chat for that provider; ?llm=default forgets it.
  const LLM_PROVIDERS = ['glm', 'groq'];
  function llmProvider() {
    const asked = new URLSearchParams(location.search).get('llm');
    if (asked === 'default') {
      try { localStorage.removeItem('jamila.llm'); } catch (e) { /* private mode */ }
    } else if (LLM_PROVIDERS.indexOf(asked) >= 0) {
      writePref('llm', asked);
    }
    const saved = readPref('llm');
    return LLM_PROVIDERS.indexOf(saved) >= 0 ? saved : null;
  }

  // ---------------------------------------------------------------- SSE transport

  function dispatchFrame(frame, onEvent) {
    let event = 'message';
    const data = [];
    frame.split('\n').forEach((line) => {
      if (line.indexOf('event:') === 0) event = line.slice(6).trim();
      else if (line.indexOf('data:') === 0) data.push(line.slice(5).replace(/^ /, ''));
    });
    if (!data.length) return;
    try {
      onEvent(event, JSON.parse(data.join('\n')));
    } catch (e) {
      /* a malformed frame is skipped */
    }
  }

  function sseTransport(base) {
    const llm = llmProvider();
    return {
      async send(request, onEvent) {
        const res = await fetch(base + '/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
          body: JSON.stringify(llm ? Object.assign({}, request, { llm: llm }) : request),
        });
        if (!res.ok || !res.body) throw new Error('HTTP ' + res.status);
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        for (;;) {
          const chunk = await reader.read();
          if (chunk.done) break;
          buffer = (buffer + decoder.decode(chunk.value, { stream: true })).replace(/\r\n/g, '\n');
          let cut;
          while ((cut = buffer.indexOf('\n\n')) >= 0) {
            dispatchFrame(buffer.slice(0, cut), onEvent);
            buffer = buffer.slice(cut + 2);
          }
        }
        if (buffer.trim()) dispatchFrame(buffer, onEvent);
      },
      async details(handle) {
        const res = await fetch(base + '/api/products/' + encodeURIComponent(handle));
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      },
    };
  }

  // ---------------------------------------------------------------- Streamlit transport

  // The custom-component protocol, without streamlit-component-lib.
  function streamlitBridge() {
    const listeners = [];
    const post = (type, data) => window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type: type }, data), '*');
    window.addEventListener('message', (e) => {
      const msg = e.data;
      if (msg && msg.type === 'streamlit:render') listeners.forEach((fn) => fn(msg.args || {}, msg));
    });
    return {
      ready: () => post('streamlit:componentReady', { apiVersion: 1 }),
      setValue: (value) => post('streamlit:setComponentValue', { value: value, dataType: 'json' }),
      setHeight: (height) => post('streamlit:setFrameHeight', { height: height }),
      onRender: (fn) => listeners.push(fn),
    };
  }

  // One request in flight at a time: each one is a Streamlit rerun, and a
  // second value sent mid-run would replace the first.
  function streamlitTransport(bridge) {
    const TIMEOUT_MS = 90000;
    const queue = [];
    let inflight = null;
    let timer = null;
    let args = {};

    function finish(fn, value) {
      clearTimeout(timer);
      const job = inflight;
      inflight = null;
      job[fn](value);
      flush();
    }

    function settle() {
      if (!inflight) return;
      const r = inflight.request;
      if (r.type === 'chat') {
        const turn = (args.turns || []).find((t) => t.id === r.id);
        if (!turn) return;
        (turn.events || []).forEach((ev) => inflight.onEvent(ev.event, ev.data));
        finish('resolve');
      } else if (r.type === 'details') {
        const details = args.details || {};
        if (!(r.id in details)) return;
        if (details[r.id]) finish('resolve', details[r.id]);
        else finish('reject', new Error('not found'));
      } else if (r.type === 'reset' && args.reset === r.id) {
        finish('resolve');
      }
    }

    function flush() {
      if (inflight || !queue.length) return;
      inflight = queue.shift();
      timer = setTimeout(() => finish('reject', new Error('timeout')), TIMEOUT_MS);
      bridge.setValue(inflight.request);
      settle();
    }

    function enqueue(request, onEvent) {
      return new Promise((resolve, reject) => {
        queue.push({ request: request, onEvent: onEvent || function () {}, resolve: resolve, reject: reject });
        flush();
      });
    }

    return {
      update(next) { args = next || {}; settle(); },
      send: (req, onEvent) => enqueue({ type: 'chat', id: newId(), message: req.message, session_id: req.session_id, locale: req.locale }, onEvent),
      details: (handle) => enqueue({ type: 'details', id: newId(), handle: handle }),
      reset: (sessionId) => enqueue({ type: 'reset', id: newId(), session_id: sessionId }),
    };
  }

  // ---------------------------------------------------------------- host page copy

  const HOST = {
    en: {
      announcement: 'Free shipping on orders above 999 EGP',
      eyebrow: 'Local demo',
      title: 'Ask Jamila',
      lede: "Majestic's assistant, wired to the chat pipeline: pre-qualification, filter extraction, retrieval and the reply. Product cards link to e-majestic.com.",
      console: 'Query console',
      language: 'Language', audience: 'Audience', theme: 'Theme',
      audiences: { auto: 'Detected from the chat', customer: 'Customer', trainee: 'Sales trainee', professional: 'Doctor or pharmacist' },
      themes: { light: 'Light', dark: 'Dark', auto: 'Follow the system' },
    },
    ar: {
      announcement: 'شحن مجاني للطلبات فوق 999 ج.م',
      eyebrow: 'نسخة تجريبية محلية',
      title: 'اسأل جميلة',
      lede: 'مساعدة Majestic متوصلة بخط المحادثة كله: فهم الرسالة، واستخراج الفلاتر، والبحث في المنتجات، والرد. كروت المنتجات بتفتح على e-majestic.com.',
      console: 'شاشة الأسئلة',
      language: 'اللغة', audience: 'المستخدم', theme: 'المظهر',
      audiences: { auto: 'حسب المحادثة', customer: 'عميل', trainee: 'متدرب مبيعات', professional: 'طبيب أو صيدلي' },
      themes: { light: 'فاتح', dark: 'غامق', auto: 'حسب الجهاز' },
    },
  };

  function renderHostCopy(locale) {
    const c = HOST[locale];
    document.querySelectorAll('[data-copy]').forEach((el) => { el.textContent = c[el.dataset.copy]; });
    document.querySelectorAll('[data-option]').forEach((el) => {
      const [group, key] = el.dataset.option.split('.');
      el.textContent = c[group][key];
    });
    document.title = c.title;
  }

  function setDocumentLanguage(locale) {
    document.documentElement.lang = locale;
    document.documentElement.dir = dirFor(locale);
  }

  function switchLanguage(widget, locale) {
    setDocumentLanguage(locale);
    widget.setDir(dirFor(locale));
    widget.setLocale(locale);
    writePref('locale', locale);
    if (!inStreamlit) {
      renderHostCopy(locale);
      const select = document.getElementById('demo-locale');
      if (select) select.value = locale;
    }
  }

  // ---------------------------------------------------------------- boot

  // Streamlit's query console shows each turn's products with args.view
  // 'cards': the widget's cards alone, sized to their content. Details come
  // with the args, so the view needs no requests back to Python.
  function cardsView(bridge) {
    let shown = null;
    let observer = null;
    return (args, theme) => {
      const locale = args.locale === 'ar' ? 'ar' : 'en';
      const key = JSON.stringify([locale, theme, (args.products || []).map((p) => p.handle)]);
      if (key === shown) return;
      shown = key;
      document.documentElement.classList.add('is-cards');
      const details = args.details || {};
      host.textContent = '';
      host.append(Jamila.ProductResults(args.products || [], {
        locale: locale, dir: dirFor(locale), theme: theme,
        details: (handle) => Promise.resolve(details[handle] || null),
      }));
      if (!observer) {
        const fit = () => bridge.setHeight(Math.ceil(host.getBoundingClientRect().height) + 4);
        observer = new ResizeObserver(fit);
        observer.observe(host);
        fit();
      }
    };
  }

  if (inStreamlit) {
    document.documentElement.classList.add('is-embedded');
    const bridge = streamlitBridge();
    const transport = streamlitTransport(bridge);
    const renderCards = cardsView(bridge);
    let widget = null;
    let audience = null;
    let theme = null;
    bridge.onRender((args, msg) => {
      const nextTheme = msg.theme && msg.theme.base === 'dark' ? 'dark' : 'light';
      if (args.view === 'cards') {
        renderCards(args, nextTheme);
        return;
      }
      if (!widget) {
        const locale = readPref('locale') || args.locale || 'en';
        setDocumentLanguage(locale);
        audience = args.audience || 'auto';
        theme = nextTheme;
        widget = Jamila.mount(host, {
          mode: 'embedded', locale: locale, dir: dirFor(locale), audience: audience, theme: theme,
          transport: transport, sessionId: args.session_id || null,
          onLocaleRequest: (next) => switchLanguage(widget, next),
        });
        (args.turns || []).forEach((t) => widget.replay(t.message, t.events));
        bridge.setHeight(args.height || 720);
      } else {
        if ((args.audience || 'auto') !== audience) widget.setAudience(audience = args.audience || 'auto');
        if (nextTheme !== theme) widget.setTheme(theme = nextTheme);
      }
      transport.update(args);
    });
    bridge.ready();
    return;
  }

  const params = new URLSearchParams(location.search);
  const pick = (key, allowed, fallback) => {
    const v = params.get(key) || readPref(key);
    return allowed.indexOf(v) >= 0 ? v : fallback;
  };
  const locale = pick('lang', ['en', 'ar'], 'en');
  const audience = pick('audience', ['auto', 'customer', 'trainee', 'professional'], 'auto');
  const theme = pick('theme', ['light', 'dark', 'auto'], 'light');

  setDocumentLanguage(locale);
  renderHostCopy(locale);
  document.documentElement.dataset.jTheme = theme;

  const widget = Jamila.mount(host, {
    mode: 'floating', locale: locale, dir: dirFor(locale), audience: audience, theme: theme,
    open: params.get('open') !== '0',
    transport: sseTransport(window.JAMILA_API_BASE || ''),
    onLocaleRequest: (next) => switchLanguage(widget, next),
  });

  const bind = (id, value, onChange) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.value = value;
    el.addEventListener('change', () => onChange(el.value));
  };
  bind('demo-locale', locale, (v) => switchLanguage(widget, v));
  bind('demo-audience', audience, (v) => { widget.setAudience(v); writePref('audience', v); });
  bind('demo-theme', theme, (v) => {
    widget.setTheme(v);
    document.documentElement.dataset.jTheme = v;
    writePref('theme', v);
  });
})();
