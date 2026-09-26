# Filter extractor: Call 3: Category

One of the 10 parallel calls in agents/filter_extractor. Output schema fields: `category`: array<enum category>.
Built at startup by agents/filter_extractor/prompts.py:

- `{{allowed}}` is replaced by this key's allowed values from data/metadata_catalog.json,
  joined with "; ". The whole `ALLOWED (...)` line is dropped when
  EXTRACTOR_ALLOWED_IN_PROMPT=false (provider enforces the schema enum).
- Everything before the first line holding `{{query}}` or `{{context}}` is the
  static system message (cacheable prefix). Those trailing lines become the
  human message, so the query always comes last.

## The prompt

```text
You extract category from a user message to the Majestic Biopharma (cosmetics/pharma) assistant.
Return the top-level categories the query is about.
The query is in English and may contain typos. Fix typos and map synonyms to the allowed values.
Rules:
- Fill when the area is stated or clearly implied (hair problem → Hair Care, deodorant → Personal Care, joint pain / supplement / wound → Health & Support).
- Several areas → several values. Greeting / off-topic → [].
- Output only the JSON object. Use values exactly as written in the list(s).
ALLOWED (category): {{allowed}}
Example:
Q: need somthing for my hair falling and a good deodrant
A: {"category": ["Hair Care", "Personal Care"]}
Q: {{query}}
```
