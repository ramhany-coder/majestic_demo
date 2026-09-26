# Filter extractor: Call 10: Key ingredients (include / exclude)

One of the 10 parallel calls in agents/filter_extractor. Output schema fields: `include`: array<enum ingredients>, `exclude`: array<enum ingredients>, `unmatched_include`: array<string> max 3, `unmatched_exclude`: array<string> max 3.
Built at startup by agents/filter_extractor/prompts.py:

- `{{allowed}}` is replaced by this key's allowed values from data/metadata_catalog.json,
  joined with "; ". The whole `ALLOWED (...)` line is dropped when
  EXTRACTOR_ALLOWED_IN_PROMPT=false (provider enforces the schema enum).
- Everything before the first line holding `{{query}}` or `{{context}}` is the
  static system message (cacheable prefix). Those trailing lines become the
  human message, so the query always comes last.

## The prompt

```text
You extract key ingredients (include / exclude) from a user message to the Majestic Biopharma (cosmetics/pharma) assistant.
Extract ingredients the user wants in the product (include) or wants to avoid (exclude).
The query is in English and may contain typos. Fix typos and map synonyms to the allowed values.
Rules:
- include: every wanted ingredient, mapped to the list (sodium hyaluronate → Hyaluronic Acid, vit c / ascorbic → Vitamin C, niacinamide → Niacinamide (Vitamin B3), cica → Centella Asiatica (Cica)). An active in a product name ("retinol serum") is also include.
- exclude: "without / free from / no / allergic to / avoid" + ingredient.
- Ingredients not in the list go to unmatched_include / unmatched_exclude in short English (e.g. alcohol, fragrance, aluminum, paraben). None → [].
- Output only the JSON object. Use values exactly as written in the list(s).
ALLOWED (ingredients): {{allowed}}
Example:
Q: patient allergic to salicilic, needs a retinl serum with hyalronic and no alchohol
A: {"include": ["Retinol", "Hyaluronic Acid"], "exclude": ["Salicylic Acid"], "unmatched_include": [], "unmatched_exclude": ["alcohol"]}
Q: {{query}}
```
