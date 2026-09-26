PY ?= python

.PHONY: test eval eval-rules eval-prequal eval-prequal-e2e catalogs catalogs-check

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

catalogs:
	$(PY) -m scripts.build_catalogs

catalogs-check:
	$(PY) -m scripts.build_catalogs --check
