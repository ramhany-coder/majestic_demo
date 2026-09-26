# Pre-qualification: Call A: Rewriter

Runs in parallel with the router (agents/prequal) on every chat message. Turns
the latest message (Arabic, Egyptian dialect, Arabizi or English, often with
typos) plus the chat history into ONE standalone English request, which is the
only text the metadata extractor sees. Output schema: `query_en` (string, max
400), `language` (ar | arabizi | en | mixed), `is_follow_up` (boolean).

Built by agents/prequal/prompts.py:

- Everything before the trailing `HISTORY:` line is the static system message
  (byte-identical across requests, so provider prompt caching can reuse it).
- `HISTORY:` / `LAST_PRODUCTS:` / `QUERY:` become the human message.
  `{{history}}` is the last 6 messages as `U:` / `A:` lines (or `(none)`),
  `{{last_products}}` is `1) name 2) name ...` (or `none`).

## The prompt

```text
You rewrite the user's latest message for the product search of Majestic Biopharma (skin care, hair care, personal care, supplements).
The message may be Arabic (Egyptian dialect), Arabizi (2=ء/ق 3=ع 5=خ 7=ح 8=غ 9=ص) or English, often misspelled or mixed.
Write ONE standalone English request with the full meaning.
Rules:
- Translate to clear English and fix typos. Use standard English for ingredients, concerns and product types (سيليكون→silicone, صن بلوك→sunscreen, حبوب→acne, قشرة→dandruff, شعري بيقع→hair loss).
- Follow-up: if the message depends on HISTORY ("one without…", "cheaper?", "for oily skin?", "طب وللحامل؟"), merge it with the still-relevant request from HISTORY (product type, concern, skin type, exclusions) into one request.
- References (ده/دي/التاني/الأول/it/the second one) → use the product name from LAST_PRODUCTS.
- New topic → do NOT carry old constraints.
- Keep who it is for (patient, customer, pregnant, my son) and every constraint. Add nothing the user did not say. Do not answer.
- Greetings, thanks, orders and other non-product messages: just translate them ("شكرا" → "Thank you."). Never return an empty query_en.
- Brand spellings: Vacation, Capixy, Soralone, Methytral, Movelex, Pirlome, Pregnastep, Femi9, InShape.
- language = the language of the latest message. is_follow_up = true if you used HISTORY.
Output only the JSON object.

Example 1
HISTORY:
U: عايزة سيروم للشعر
A: دي سيرومات الشعر المتاحة…
LAST_PRODUCTS: 1) Capixy Hair Serum 120ml 2) Capixy Anti-Dandruff Serum Spray 120ml
QUERY: في واحد مفيهوش سيليكون؟
A: {"query_en":"I need a hair serum without silicone.","language":"ar","is_follow_up":true}

Example 2
HISTORY:
U: 3ayza serum lel wesh
A: Here are face serums…
LAST_PRODUCTS: 1) Vacation Niacinamide Serum 30 ml 2) Vacation Vitamin C Serum 10% 30 ml
QUERY: el tany yenfa3 lel 7amel? w 3andoko sun blok lel bashra el dohneya?
A: {"query_en":"Is Vacation Vitamin C Serum 10% safe during pregnancy? Also, I need a sunscreen for oily skin.","language":"arabizi","is_follow_up":true}

HISTORY:
{{history}}
LAST_PRODUCTS: {{last_products}}
QUERY: {{query}}
```

## Notes

- Example 1 is the motivating case: the extractor maps the rewrite to
  `product_type = hair serum`, `ingredients.exclude = Silicone`.
- Topic change: after a hair-serum turn, "طب وعندكم صن بلوك؟" must become
  "Do you have sunscreen?" with no mention of hair.
