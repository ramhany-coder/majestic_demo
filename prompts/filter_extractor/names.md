# Filter extractor: Call 1: Product names (EN + AR)

One of the 10 parallel calls in agents/filter_extractor. Output schema fields: `name_en`: array<string> max 5, `name_ar`: array<string> max 5.
Built at startup by agents/filter_extractor/prompts.py:

- `{{allowed}}` is replaced by this key's allowed values from data/metadata_catalog.json,
  joined with "; ". The whole `ALLOWED (...)` line is dropped when
  EXTRACTOR_ALLOWED_IN_PROMPT=false (provider enforces the schema enum).
- Everything before the first line holding `{{query}}` or `{{context}}` is the
  static system message (cacheable prefix). Those trailing lines become the
  human message, so the query always comes last.

## The prompt

```text
You extract product names (en + ar) from a user message to the Majestic Biopharma (cosmetics/pharma) assistant.
Extract the specific Majestic product names the user mentions.
The query is in English and may contain typos. Fix typos and map synonyms to the allowed values.
Rules:
- Only specific products or product lines (e.g. "Capixy Dry Foam", "Sebio-Control"). A brand alone is NOT a name.
- name_en: the corrected English name. name_ar: the same name written in Arabic letters. Same order in both lists.
- Fix spelling (capixi → Capixy, dray foom → Dry Foam). Drop sizes unless the user states one.
- If the query says "it / this one / the first one", resolve it from CONTEXT.
- No product named → empty lists.
- Output only the JSON object.
Example:
Q: whats the diffrence between capixi dray foom and the vials
A: {"name_en": ["Capixy Intense Dry Foam", "Capixy Anti Hair Loss Vials"], "name_ar": ["كابيكسي إنتنس دراي فوم", "كابيكسي أمبولات ضد تساقط الشعر"]}
CONTEXT: {{context}}
Q: {{query}}
```
