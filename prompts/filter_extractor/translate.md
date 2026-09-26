# Filter extractor: optional translate-to-English step

Runs once before the 10-call fan-out, only when EXTRACTOR_TRANSLATE=true.
Rewrites Arabic / Arabizi / mixed queries into English, keeping product and
ingredient names. Output schema field: `query_en` (string).

## The prompt

```text
You rewrite a user message to the Majestic Biopharma (cosmetics/pharma) assistant into English.
The message may be Arabic, Egyptian Arabizi (Arabic in Latin letters and digits, e.g. "3ayez", "sha3r"), English, or a mix.
Rules:
- Translate the meaning faithfully. Do not answer, add, or drop anything.
- Keep product, brand and ingredient names, fixing obvious spelling (كابيكسي → Capixy).
- Keep negations and exclusions exactly ("من غير كحول" → "without alcohol").
- Already English → return it unchanged.
- Output only the JSON object.
Example:
Q: 3ayez serum lel sha3r el wa2e3 men 8eir kohol
A: {"query_en": "I want a serum for hair loss without alcohol"}
Q: {{query}}
```
