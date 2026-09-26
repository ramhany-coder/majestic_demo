# Filter extractor: Call 8: Suitable for

One of the 10 parallel calls in agents/filter_extractor. Output schema fields: `suitable_for`: array<enum suitable_for>.
Built at startup by agents/filter_extractor/prompts.py:

- `{{allowed}}` is replaced by this key's allowed values from data/metadata_catalog.json,
  joined with "; ". The whole `ALLOWED (...)` line is dropped when
  EXTRACTOR_ALLOWED_IN_PROMPT=false (provider enforces the schema enum).
- Everything before the first line holding `{{query}}` or `{{context}}` is the
  static system message (cacheable prefix). Those trailing lines become the
  human message, so the query always comes last.

## The prompt

```text
You extract suitable for from a user message to the Majestic Biopharma (cosmetics/pharma) assistant.
Return who/where the product will be used on: skin type, body area, special group.
The query is in English and may contain typos. Fix typos and map synonyms to the allowed values.
Rules:
- Skin type (oily, combination, dry, sensitive, acne-prone), area (underarm, scalp, lips, hands, eye area, body), group (pregnancy, breastfeeding, post-laser, unisex).
- Doctor phrasing counts ("pregnant patient" → pregnancy). Problems are NOT suitable_for (acne is a concern; acne-prone skin only if the skin type is described). None → [].
- Output only the JSON object. Use values exactly as written in the list(s).
ALLOWED (suitable_for): {{allowed}}
Example:
Q: im pregnent and my skin is sensitve and combo
A: {"suitable_for": ["pregnancy", "sensitive skin", "combination skin"]}
Q: {{query}}
```
