# Filter extractor: Call 5: Product type

One of the 10 parallel calls in agents/filter_extractor. Output schema fields: `product_type`: array<enum product_type>.
Built at startup by agents/filter_extractor/prompts.py:

- `{{allowed}}` is replaced by this key's allowed values from data/metadata_catalog.json,
  joined with "; ". The whole `ALLOWED (...)` line is dropped when
  EXTRACTOR_ALLOWED_IN_PROMPT=false (provider enforces the schema enum).
- Everything before the first line holding `{{query}}` or `{{context}}` is the
  static system message (cacheable prefix). Those trailing lines become the
  human message, so the query always comes last.

## The prompt

```text
You extract product type from a user message to the Majestic Biopharma (cosmetics/pharma) assistant.
Return the specific product types the user wants.
The query is in English and may contain typos. Fix typos and map synonyms to the allowed values.
Rules:
- Serums: face serum (face / vitamin c / niacinamide / retinol), hair serum, eye serum (eyes / dark circles), lash serum. Plain "serum" with no area → [] (the form call handles it).
- sun block / spf → sunscreen. Anti-sweat → antiperspirant, odor → deodorant.
- Only types the user asks for, not types you would recommend. None → [].
- Output only the JSON object. Use values exactly as written in the list(s).
ALLOWED (product_type): {{allowed}}
Example:
Q: i want a vitamn c seerum for my face and a sun blok
A: {"product_type": ["face serum", "sunscreen"]}
Q: {{query}}
```
