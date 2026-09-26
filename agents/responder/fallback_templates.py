"""Template answers built in code from the data, in the reply language. Used
when every R1 route fails, or when the price check rejects R1's answer.
Never empty.
"""

from typing import List

from agents.responder.cards import display_name
from agents.responder.context import ResponderContext

T = {
    "ar": {
        "how_to_use": "طريقة استخدام {name}:",
        "price": "{name} سعره {price} ج.م",
        "was": " بدل {was} ج.م",
        "offer": {"buy1get1": " (عرض اشتري 1 وخد 1)", "buy2get1": " (عرض اشتري 2 وخد 1)"},
        "sold_out": " (مش متاح حاليا)",
        "no_match": "مش لاقية تطابق كامل، دي أقرب الاختيارات:",
        "no_match_empty": "مش لاقية منتج مطابق للطلب. ممكن توضيح أكتر، أو التواصل مع خدمة عملاء ماجستيك.",
        "warnings": "تحذيرات {name} المكتوبة:",
        "safety": "مفيش معلومة كافية عن ده في بيانات {name}. الأفضل استشارة الدكتور أو الصيدلي قبل الاستخدام.",
        "products": "دي المنتجات المناسبة:",
        "greeting": "أهلا، أقدر أساعد في البشرة أو الشعر أو أي منتج من ماجستيك.",
        "out_of_scope": "للطلبات والتوصيل والمرتجعات، خدمة عملاء ماجستيك هتساعد في ده.",
        "apology": "معلش، مقدرتش أكتب الرد دلوقتي. ممكن المحاولة تاني أو التواصل مع خدمة عملاء ماجستيك.",
    },
    "arabizi": {
        "how_to_use": "Tare2et este5dam {name}:",
        "price": "{name} se3ro {price} EGP",
        "was": " badal {was} EGP",
        "offer": {"buy1get1": " (3ard eshtery 1 w 5od 1)", "buy2get1": " (3ard eshtery 2 w 5od 1)"},
        "sold_out": " (mesh metah delwa2ty)",
        "no_match": "Mesh la2ya tatabo2 kamel, dy a2rab el e5tyarat:",
        "no_match_empty": "Mesh la2ya montag motabe2. Momken tawde7 aktar, aw el tawasol ma3 5edmet 3omala2 Majestic.",
        "warnings": "Ta7zeerat {name} el maktouba:",
        "safety": "Mafeesh ma3loma kefaya 3an da fe bayanat {name}. El afdal estesharet el doktor aw el saydaly.",
        "products": "Dy el montagat el monasba:",
        "greeting": "Ahlan, a2dar asa3ed fel bashra aw el sha3r aw ay montag men Majestic.",
        "out_of_scope": "Lel talabat wel tawseel wel morta3at, 5edmet 3omala2 Majestic hatsa3ed fe da.",
        "apology": "Ma3lesh, ma2dertesh akteb el rad delwa2ty. Momken el mo7awla tany aw el tawasol ma3 5edmet 3omala2 Majestic.",
    },
    "en": {
        "how_to_use": "How to use {name}:",
        "price": "{name} costs {price} EGP",
        "was": " (was {was} EGP)",
        "offer": {"buy1get1": " (buy 1 get 1 offer)", "buy2get1": " (buy 2 get 1 offer)"},
        "sold_out": " (currently out of stock)",
        "no_match": "I couldn't find an exact match. These are the closest options:",
        "no_match_empty": "I couldn't find a product matching your request. Could you tell me more, or contact "
                          "Majestic customer service?",
        "warnings": "The listed warnings for {name}:",
        "safety": "This isn't stated in the data for {name}. Please ask a doctor or pharmacist before use.",
        "products": "Here are the matching products:",
        "greeting": "Hello. I can help with skin, hair or any Majestic product.",
        "out_of_scope": "For orders, delivery and returns, please contact Majestic customer service.",
        "apology": "Sorry, I couldn't write an answer right now. Please try again or contact Majestic customer "
                   "service.",
    },
}


def _price(value) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def _price_line(p: dict, t: dict, reply_lang: str) -> str:
    line = t["price"].format(name=display_name(p, reply_lang), price=_price(p.get("price")))
    if p.get("compare_at_price") and p.get("compare_at_price") != p.get("price"):
        line += t["was"].format(was=_price(p["compare_at_price"]))
    line += t["offer"].get(p.get("promotion") or "", "")
    if not p.get("available", True):
        line += t["sold_out"]
    return line


def _names(products: List[dict], reply_lang: str) -> str:
    return "\n".join(f"- {display_name(p, reply_lang)}" for p in products)


def template_answer(ctx: ResponderContext, products: List[dict], reply_lang: str) -> str:
    t = T.get(reply_lang, T["en"])
    if ctx.mode == "no_match":
        return f"{t['no_match']}\n{_names(products, reply_lang)}" if products else t["no_match_empty"]
    if ctx.retrieval is None or ctx.intent in ("greeting", "out_of_scope"):
        return t["greeting"] if ctx.intent == "greeting" else t["out_of_scope"]
    if products:
        first = products[0]
        if ctx.intent == "how_to_use" and first.get("how_to_use"):
            steps = "\n".join(f"{i}. {s}" for i, s in enumerate(first["how_to_use"], 1))
            return f"{t['how_to_use'].format(name=display_name(first, reply_lang))}\n{steps}"
        if ctx.intent == "price_offer":
            return "\n".join(_price_line(p, t, reply_lang) for p in products[:3])
        if ctx.intent == "safety":
            if first.get("warnings"):
                warnings = "\n".join(f"- {w}" for w in first["warnings"])
                return f"{t['warnings'].format(name=display_name(first, reply_lang))}\n{warnings}"
            return t["safety"].format(name=display_name(first, reply_lang))
        if ctx.intent in ("find_products", "refine_products"):
            return f"{t['products']}\n{_names(products, reply_lang)}"
    return t["apology"]
