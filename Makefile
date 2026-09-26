PY ?= python

.PHONY: serve streamlit web web-check test eval eval-rules eval-prequal eval-prequal-e2e eval-retrieval smoke catalogs catalogs-check

# API server: the query console at http://127.0.0.1:8000/, the widget at /index.html (see web/README.md).
serve:
	$(PY) -m uvicorn api.app:app --reload

# Streamlit: the query console (default) or the widget, with the pipeline in-process.
streamlit:
	$(PY) -m streamlit run streamlit_app.py

# Regenerate web/tokens.css and the bundle's icons; check token contrast.
web:
	$(PY) -m scripts.build_web

web-check:
	$(PY) -m scripts.build_web --check

test:
	$(PY) -m pytest -q

# Live run against the configured LLM routes (paced for the Groq free tier).
eval:
	$(PY) -u -m scripts.eval_extractor --mode llm

# Offline: rule-based fallback only, no API calls.
eval-rules:
	$(PY) -u -m scripts.eval_extractor --mode rules

# Pre-qualification (rewriter + router) on tests/prequal_eval.jsonl: live prequal,
# extractor stage on the offline rules. The -e2e target runs the live extractor too.
eval-prequal:
	$(PY) -u -m scripts.eval_prequal

eval-prequal-e2e:
	$(PY) -u -m scripts.eval_prequal --extractor llm

# Retrieval on tests/retrieval_eval.jsonl (precision@k, recall@k, MRR, latency) plus
# the name typo test. No LLM; uses the local embedding model.
eval-retrieval:
	$(PY) -u -m scripts.eval_retrieval

# Live end-to-end smoke test: raw Arabic / Arabizi messages through the whole pipeline.
smoke:
	$(PY) -u -m scripts.smoke_e2e

catalogs:
	$(PY) -m scripts.build_catalogs

catalogs-check:
	$(PY) -m scripts.build_catalogs --check
