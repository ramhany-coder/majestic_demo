/* Query console (web/console.html): sends each query to POST /chat and shows
   the reply, the matched products as the widget's cards (Jamila.ProductResults;
   hidden while "Show matched products" is off), and what the pipeline did.
   The session keeps follow-ups in context until "New session". */
(function () {
  'use strict';

  const log = document.getElementById('log');
  const form = document.getElementById('ask');
  const input = document.getElementById('q');
  const sendButton = document.getElementById('send');
  const OPTIONS = { products: document.getElementById('opt-products'), details: document.getElementById('opt-details') };
  const STAGES = {
    understanding: 'Reading the query',
    searching: 'Searching the catalogue',
    writing: 'Writing the reply',
  };
  const DISCLAIMER = 'This information is for awareness and is not a substitute for consulting a doctor. · المعلومات للتوعية ومش بديلة عن استشارة الطبيب';
  let session = newId();
  let busy = false;
  // Hidden LLM switch, as in the widget: console.html?llm=groq or ?llm=glm.
  const LLM = ['glm', 'groq'].indexOf(new URLSearchParams(location.search).get('llm')) >= 0
    ? new URLSearchParams(location.search).get('llm') : null;

  function newId() {
    return (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : 's-' + Date.now().toString(36) + Math.random().toString(36).slice(2);
  }

  function h(tag, props) {
    const el = document.createElement(tag);
    Object.entries(props || {}).forEach(([k, v]) => {
      if (v == null || v === false) return;
      if (k === 'class') el.className = v;
      else el.setAttribute(k, v);
    });
    [].slice.call(arguments, 2).flat(Infinity).forEach((c) => {
      if (c != null && c !== false) el.append(c);
    });
    return el;
  }

  function scrollToEnd() {
    requestAnimationFrame(() => { log.scrollTop = log.scrollHeight; });
  }

  // ---- options: per-viewer, remembered in this browser ----

  function applyOptions() {
    document.body.classList.toggle('show-products', OPTIONS.products.checked);
    document.body.classList.toggle('show-details', OPTIONS.details.checked);
  }

  Object.entries(OPTIONS).forEach(([key, box]) => {
    const storageKey = 'jamila.console.v2.' + key;
    try {
      const saved = localStorage.getItem(storageKey);
      if (saved !== null) box.checked = saved === '1';
    } catch (e) { /* storage unavailable */ }
    box.addEventListener('change', () => {
      try { localStorage.setItem(storageKey, box.checked ? '1' : '0'); } catch (e) { /* ignore */ }
      applyOptions();
    });
  });
  applyOptions();

  // ---- the /chat stream ----

  function dispatchFrame(frame, onEvent) {
    let event = 'message';
    const data = [];
    frame.split('\n').forEach((line) => {
      if (line.indexOf('event:') === 0) event = line.slice(6).trim();
      else if (line.indexOf('data:') === 0) data.push(line.slice(5).replace(/^ /, ''));
    });
    if (!data.length) return;
    let payload;
    try { payload = JSON.parse(data.join('\n')); } catch (e) { return; }
    onEvent(event, payload);
  }

  async function stream(query, onEvent) {
    const res = await fetch('chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify(Object.assign({ message: query, session_id: session, locale: 'en' }, LLM ? { llm: LLM } : {})),
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
  }

  // ---- one turn ----

  function formatFilters(applied) {
    const parts = Object.entries(applied || {}).filter(([, v]) => v && v.length).map(([k, v]) => k + ': ' + v.join(', '));
    return parts.length ? parts.join(' · ') : 'none';
  }

  function renderDetails(done) {
    const p = done.pipeline || {};
    const timings = Object.entries(done.timings_ms || {}).map(([k, v]) => k + ' ' + Math.round(v) + ' ms').join(' · ');
    const rows = [
      ['Rewritten request', p.query_en],
      ['Route', [done.route, p.needs_retrieval ? 'retrieval' : 'no retrieval', p.is_follow_up ? 'follow-up' : null].filter(Boolean).join(' · ')],
      ['Model status', Object.entries(p.model_status || {}).map(([k, v]) => k + ' ' + v).join(' · ') || 'n/a'],
      ['Applied filters', p.retrieval_ran ? formatFilters(p.applied_filters) : 'retrieval did not run'],
      p.relaxed_keys && p.relaxed_keys.length ? ['Relaxed filters', p.relaxed_keys.join(', ')] : null,
      p.retrieval_ran ? ['Candidates', String(p.total_candidates)] : null,
      ['Timings', timings],
    ].filter(Boolean);
    return h('details', { class: 'c-details' },
      h('summary', null, 'Pipeline details'),
      h('dl', null, rows.map(([k, v]) => [h('dt', null, k), h('dd', { dir: 'auto' }, v || '')])));
  }

  function renderProducts(data, language) {
    const items = data.items || [];
    const locale = language === 'ar' || language === 'mixed' ? 'ar' : 'en';
    const n = items.length;
    return [
      h('p', { class: 'c-count' }, n === 1 ? '1 product matched' : n + ' products matched',
        ' (turn on "Show matched products" to see them)'),
      Jamila.ProductResults(items, {
        locale: locale,
        dir: locale === 'ar' ? 'rtl' : 'ltr',
        details: (handle) => fetch('api/products/' + encodeURIComponent(handle)).then((r) => (r.ok ? r.json() : null)),
      }),
    ];
  }

  function setBusy(value) {
    busy = value;
    sendButton.disabled = value;
  }

  async function ask(query) {
    const empty = log.querySelector('.c-empty');
    if (empty) empty.remove();
    const status = h('p', { class: 'c-status' }, STAGES.understanding);
    const answer = h('div', { class: 'c-answer' }, status);
    log.append(h('article', { class: 'c-turn' }, h('p', { class: 'c-query', dir: 'auto' }, query), answer));
    scrollToEnd();
    setBusy(true);
    let language = 'en';
    let replyEl = null;
    let replied = false;
    let failed = false;
    try {
      await stream(query, (event, data) => {
        if (event === 'status') {
          status.textContent = STAGES[data.stage] || '';
        } else if (event === 'message') {
          replied = true;
          language = data.language || 'en';
          replyEl = h('p', { class: 'c-reply', dir: 'auto' }, data.text || '');
          status.before(replyEl);
          if (data.health) status.before(h('p', { class: 'c-disclaimer' }, DISCLAIMER));
        } else if (event === 'products') {
          status.before(...renderProducts(data, language));
        } else if (event === 'done') {
          const seconds = ((data.timings_ms || {}).total || 0) / 1000;
          const meta = h('p', { class: 'c-meta' },
            [data.path, data.intent, data.persona, language].filter(Boolean).join(' · ') + ' · ' + seconds.toFixed(2) + ' s');
          if (replyEl) replyEl.after(meta);
          else status.before(meta);
          answer.append(renderDetails(data));
        } else if (event === 'error') {
          failed = true;
        }
      });
    } catch (e) {
      failed = true;
      status.textContent = 'The pipeline failed: ' + e.message;
    }
    if (failed || !replied) {
      status.className = 'c-error';
      status.setAttribute('role', 'alert');
      if (!status.textContent.startsWith('The pipeline failed')) status.textContent = 'The pipeline failed. See the server log.';
    } else {
      status.remove();
    }
    setBusy(false);
    scrollToEnd();
  }

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query || busy) return;
    input.value = '';
    ask(query);
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  document.getElementById('new').addEventListener('click', () => {
    session = newId();
    log.textContent = '';
    log.append(h('p', { class: 'c-empty' }, 'New session. Earlier queries are no longer used as context.'));
    input.focus();
  });

  input.focus();
})();
