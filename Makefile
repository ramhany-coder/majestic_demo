PY ?= python

.PHONY: test eval eval-rules catalogs catalogs-check

test:
	$(PY) -m pytest -q

# Live run against the configured LLM routes (paced for the Groq free tier).
eval:
	$(PY) -u -m scripts.eval_extractor --mode llm

# Offline: rule-based fallback only, no API calls.
eval-rules:
	$(PY) -u -m scripts.eval_extractor --mode rules

catalogs:
	$(PY) -m scripts.build_catalogs

catalogs-check:
	$(PY) -m scripts.build_catalogs --check
