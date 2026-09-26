# Filter extractor: Call 4: Product group

One of the 10 parallel calls in agents/filter_extractor. Output schema fields: `product_group`: array<enum product_group>.
Built at startup by agents/filter_extractor/prompts.py:

- `{{allowed}}` is replaced by this key's allowed values from data/metadata_catalog.json,
  joined with "; ". The whole `ALLOWED (...)` line is dropped when
  EXTRACTOR_ALLOWED_IN_PROMPT=false (provider enforces the schema enum).
- Everything before the first line holding `{{query}}` or `{{context}}` is the
  static system message (cacheable prefix). Those trailing lines become the
  human message, so the query always comes last.

## The prompt

```text
You extract product group from a user message to the Majestic Biopharma (cosmetics/pharma) assistant.
Return the product groups the query is about.
The query is in English and may contain typos. Fix typos and map synonyms to the allowed values.
Rules:
- Map body area or need to a group (dark circles / lashes → Eye & Lash Care, lips → Lip Care, sun block → Sun Care, wounds / scars → Skin Repair & Healing).
- Bundles & Offers only if the user asks for a bundle / offer / set. None → [].
- Output only the JSON object. Use values exactly as written in the list(s).
ALLOWED (product_group): {{allowed}}
Example:
Q: a cream for dark circels and something for chapped lipps
A: {"product_group": ["Eye & Lash Care", "Lip Care"]}
Q: {{query}}
```
