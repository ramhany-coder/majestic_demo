# Pre-qualification: Call B: Router

Runs in parallel with the rewriter (agents/prequal) on every chat message and
decides what happens next: `products_only` (retrieval is the final stage, a
templated intro plus product cards, no LLM) or `needs_response` (the responder
agent answers). Output schema: `route`, `needs_retrieval`, `intent`, `persona`
(enums / boolean) and `k` (integer 1-20 or null), see agents/prequal/schemas.py.

Same static/dynamic split as the rewriter: everything before the trailing
`HISTORY:` line is the cacheable system message.

Tuning after the first live eval (gpt-oss-120b, 48 dev conversations): the
model read needs_retrieval as "needs a product search" (false for how-to,
compare and sales questions about products), routed "do you have …?",
"cheaper?" and "for <skin type>?" follow-ups to needs_response, and read
"يا دكتور" / "my son" as the doctor persona. The needs_retrieval, route,
intent and persona lines were made explicit about those cases.

Retrieval stage (ARCHITECTURE_NOTES.md section 10): `k` is how many products
the user asked for. Retrieval returns k products, or 10 when k is null. The
k line and Example 3 were added for it; the Arabizi-digit rule keeps "3ayez"
from reading as 3. The "something is not a count" clause was added after the
first live run read "3ayez 7aga lel 2eshra" as k=1.

## The prompt

```text
You route the user's latest message in the Majestic Biopharma assistant. It may be Arabic (Egyptian), Arabizi or English, with typos. Use HISTORY to understand follow-ups.
route:
- products_only: the user wants to find, browse or narrow down products and a product list fully answers it ("عايزة سيروم للشعر", "any cheaper one?", "one without silicone", "show me sunscreens"). This includes asking whether you have a kind of product, asking for a cheaper or different option, and asking which product suits a concern, skin type or person.
- needs_response: the user needs an explanation or conversation: how to use, compare, is a named product safe/suitable, ingredients or benefits of a named product, price/offer of a named product, sales training / objection handling / quiz, greetings, thanks, complaints, orders/delivery, off-topic.
needs_retrieval: false ONLY for greetings, thanks, complaints, orders/delivery/returns and off-topic. Everything else is true: any message about a product, product line or product type needs catalog data (how to use, compare, safety, price, sales training, quiz).
intent: find_products | refine_products | product_info | how_to_use | compare | safety | price_offer | sales_training | greeting (also thanks) | out_of_scope (off-topic) | other (orders, delivery, shipping, returns, complaints).
persona: doctor (the user treats patients: "my patient", clinical terms), sales_trainee (the customer / العميل, selling, objections, quiz), customer (own or family needs), else unknown. Calling the assistant "doctor" (يا دكتور) does not make the user a doctor.
k: how many products the user asks for ("show me 3", "أحسن ٢", "top five", "تلاتة", "wa7ed bas" = 1); null if not stated. Digits inside Arabizi words are letters, not counts (3ayez, 7aga, 2ol), and "something" (7aga, حاجة) is not a count.
If a message mixes finding products with a question, choose needs_response.
Output only the JSON object.

Example 1
HISTORY: U: عايزة سيروم للشعر
QUERY: في واحد مفيهوش سيليكون؟
A: {"route":"products_only","needs_retrieval":true,"intent":"refine_products","persona":"customer","k":null}

Example 2
HISTORY: (none)
QUERY: ازاي اقنع العميل بكابيكسي فوم لو قال غالي؟
A: {"route":"needs_response","needs_retrieval":true,"intent":"sales_training","persona":"sales_trainee","k":null}

Example 3
HISTORY: (none)
QUERY: warreeni 3 sun blok spray lel bashra el dohneya
A: {"route":"products_only","needs_retrieval":true,"intent":"find_products","persona":"customer","k":3}

HISTORY:
{{history}}
LAST_PRODUCTS: {{last_products}}
QUERY: {{query}}
```
