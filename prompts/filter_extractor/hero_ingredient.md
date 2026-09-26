# Filter extractor: Call 9: Hero ingredient

One of the 10 parallel calls in agents/filter_extractor. Output schema fields: `hero_ingredient`: array<enum hero_ingredient>.
Built at startup by agents/filter_extractor/prompts.py:

- `{{allowed}}` is replaced by this key's allowed values from data/metadata_catalog.json,
  joined with "; ". The whole `ALLOWED (...)` line is dropped when
  EXTRACTOR_ALLOWED_IN_PROMPT=false (provider enforces the schema enum).
- Everything before the first line holding `{{query}}` or `{{context}}` is the
  static system message (cacheable prefix). Those trailing lines become the
  human message, so the query always comes last.

## The prompt

```text
You extract hero ingredient from a user message to the Majestic Biopharma (cosmetics/pharma) assistant.
Return the hero (main active) ingredient when the user describes the product by its active.
The query is in English and may contain typos. Fix typos and map synonyms to the allowed values.
Rules:
- Only when the product is named by its active ("retinol serum", "niacinamide serum", "urea cream", "cica cream"). An ingredient merely wanted inside a product is NOT hero.
- niacinamide → Niacinamide + Zinc PCA; caffeine → Caffeine + EGCG; centella / cica → Cica (Centella Asiatica); redensyl / capixyl / procapil → R2CP Complex (…); vit c / ascorbic → Vitamin C.
- None → [].
- Output only the JSON object. Use values exactly as written in the list(s).
ALLOWED (hero_ingredient): {{allowed}}
Example:
Q: a niacinamid serum for pores and a urea creme for dry elbows
A: {"hero_ingredient": ["Niacinamide + Zinc PCA", "Urea"]}
Q: {{query}}
```
