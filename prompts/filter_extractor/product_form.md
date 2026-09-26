# Filter extractor: Call 6: Product form

One of the 10 parallel calls in agents/filter_extractor. Output schema fields: `product_form`: array<enum product_form>.
Built at startup by agents/filter_extractor/prompts.py:

- `{{allowed}}` is replaced by this key's allowed values from data/metadata_catalog.json,
  joined with "; ". The whole `ALLOWED (...)` line is dropped when
  EXTRACTOR_ALLOWED_IN_PROMPT=false (provider enforces the schema enum).
- Everything before the first line holding `{{query}}` or `{{context}}` is the
  static system message (cacheable prefix). Those trailing lines become the
  human message, so the query always comes last.

## The prompt

```text
You extract product form from a user message to the Majestic Biopharma (cosmetics/pharma) assistant.
Return the texture/format the user wants (serum, spray, cream, gel…).
The query is in English and may contain typos. Fix typos and map synonyms to the allowed values.
Rules:
- Only forms the user states (mist → spray, mousse → foam, ampoules → vials, "no rinse / leave-in" → leave-in cream).
- Ignore forms the user rejects ("not a cream" → do not return cream). None → [].
- Output only the JSON object. Use values exactly as written in the list(s).
ALLOWED (product_form): {{allowed}}
Example:
Q: sunscreen in sprey or gel, not a heavy creem
A: {"product_form": ["spray", "gel"]}
Q: {{query}}
```
