# Responder: R2, the sales training card (structured)

Runs in parallel with R1, only when intent = sales_training
(agents/responder/sales_card.py). Output schema: `SALES_CARD_SCHEMA` in
that file; the output is validated in code and dropped when invalid.

## The prompt

```text
Create a training card for a Majestic sales trainee in {{reply_language}}, using ONLY PRODUCT DATA.
card_type = "objection" if the message is about a customer objection (price, doubt, competitor), else "quiz".
quiz: 3 multiple-choice questions about benefits, hero ingredient, who it suits, usage or offer; one correct option each; short explanation.
objection: restate the objection, 3 talking points (Acknowledge → Reframe value → Close), one suggested reply to the customer.
For card_type "quiz" set objection to null; for "objection" set quiz to [].
Never invent facts. Output only the JSON.
PRODUCT DATA:
{{product_json_lines}}
MESSAGE: {{query_original}}
```

## Notes

- The "set objection to null / quiz to []" line was added: the schema
  requires both keys, and JSON-mode providers need to be told what the
  unused one holds.
