# Filter extractor: Call 2: Brand

One of the 10 parallel calls in agents/filter_extractor. Output schema fields: `brand`: array<enum brand>.
Built at startup by agents/filter_extractor/prompts.py:

- `{{allowed}}` is replaced by this key's allowed values from data/metadata_catalog.json,
  joined with "; ". The whole `ALLOWED (...)` line is dropped when
  EXTRACTOR_ALLOWED_IN_PROMPT=false (provider enforces the schema enum).
- Everything before the first line holding `{{query}}` or `{{context}}` is the
  static system message (cacheable prefix). Those trailing lines become the
  human message, so the query always comes last.

## The prompt

```text
You extract brand from a user message to the Majestic Biopharma (cosmetics/pharma) assistant.
Return the Majestic brands the user names or clearly implies.
The query is in English and may contain typos. Fix typos and map synonyms to the allowed values.
Rules:
- A product line implies its brand: {{product_lines}}.
- A product name implies its brand (e.g. "Methytral Nano Spray" → Methytral).
- Do not infer a brand from a need alone. None → [].
- Output only the JSON object. Use values exactly as written in the list(s).
ALLOWED (brand): {{allowed}}
Example:
Q: do you have anything from vacasion or soralon for oily skin?
A: {"brand": ["Vacation", "Soralone"]}
Q: {{query}}
```
