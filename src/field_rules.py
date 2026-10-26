"""Общий контракт извлечения, смысловой проверки и разрешённых вариантов значений.

Здесь нет эталонов или номеров кейсов. Строгая JSON-метрика остаётся отдельной:
эквивалентность ограничена явно разрешёнными вариантами двух полей перевода.
"""

# Под вопросом
PERMITTED_ALIASES = {
    ("transfer_draft", "destination_bank"): (("Сбер", "Сбера", "Сбербанк"),),
    ("transfer_draft", "source_account_kind"): (("расчётный счёт", "расчётный"),),
}

TRANSFER_ALIAS_RULES = {
    name: f"Permitted equivalent forms for {document_type}.{name}: "
    + "; ".join(" / ".join(group) for group in groups)
    + ". Do not flag a permitted form as unsupported or incorrect."
    for (document_type, name), groups in PERMITTED_ALIASES.items()
}

TRANSFER_RECIPIENT_RULE = (
    "recipient_relation='self' when the user explicitly requests a transfer to themselves ('себе') "
    "or their own card. Missing card/bank details do not negate self. The selected document type "
    "alone is not evidence; another person's card does not support self. Never invent card/bank "
    "details from this relation."
)

ITEMS_PRESENCE_RULE = (
    "Selected billed goods/services require one line each and items.status=fillable. An item with "
    "only a supported description is fillable. Known attributes need user evidence or an "
    "unambiguous calculation; leave all other attributes null. Missing attributes do not make "
    "items missing/ambiguous. Do not create an item from only a total amount, a recipient name or "
    "a contract reference. Without billed goods/services or other item data, items is missing."
)

ITEMS_SELECTION_RULE = (
    "Only unambiguously selected goods/services are active billed items. "
    "For 'стол и либо стул, либо кресло — ещё решу', keep only стол; "
    "unchosen alternatives are not omitted active items. If none are selected, "
    "the expected items state is status=ambiguous, value=null with evidence. "
    "For selected items with unresolved attributes, keep status=fillable and only those "
    "attributes null. Later explicit choices/corrections resolve the alternatives."
)

ITEMS_EXAMPLES = (
    "Examples only, not actual input. 'Выставь счёт за консультацию': description=консультация, "
    "other attributes null. 'Счёт на 5000 рублей': amount_rub=5000, items missing. 'Счёт за "
    "консультационные услуги на 5000 рублей': description=консультационная услуга, "
    "subject=консультационные услуги, amount_rub=5000; other item attributes null. 'Абрикосы: 100 "
    "рублей за 5 штук': quantity=5, unit=штук, unit_price_rub=100/5. 'Стулья: по 3 штуки для каждого из 4 "
    "кабинетов, всего 12000 рублей': quantity=3*4, unit_price_rub=12000/(3*4). 'Яблоки и груши: всего 5 "
    "штук на 100 рублей': amount_rub=100; two lines яблоко/груша, each unit=штук and "
    "quantity/unit_price_rub/calculations=null; subject missing. Do not allocate the combined count or "
    "total between these goods."
)

RECIPIENT_NAME_RULE = (
    "Use nominative case for a person name only when unambiguous. Preserve "
    "company names, quoted text and name components; invent none."
)

BASIS_NUMBER_RULE = (
    "Copy the identifier without № or labels договор/номер; preserve zeros, letters, slashes and "
    "hyphens."
)

VAT_MODE_RULE = (
    "Distinguish no_vat from a stated 0% rate. 'Без НДС' means vat_mode='no_vat' and "
    "vat_rate_percent is missing; 'НДС 0%' means vat_mode='rate' and vat_rate_percent=0. A rate "
    "alone does not authorize adding VAT to the total."
)

VAT_RATE_RULE = (
    "Only an explicitly stated percentage supports a VAT rate; do not derive it from tax and "
    "base amounts. If no rate is stated, vat_rate_percent is missing. For this field "
    "calculation=null."
)

ITEM_DESCRIPTION_RULE = (
    "For item descriptions only, use singular nominative when unambiguous; never infer quantity=1. "
    "Preserve meaning, characteristics, established names, names in quotation marks, plural-only "
    "nouns (ножницы) and ambiguous forms. Do not singularize subject or change evidence."
)

ITEM_DESCRIPTION_SEMANTIC_RULE = (
    "Check product/service identity and stated characteristics only. Singular nominative "
    "normalization ('стулья' -> 'стул') preserves identity; number/case changes alone are not errors. "
    "Do not judge quantity, unit or price. "
    "A different product, invented characteristics or a changed quoted product name is an error."
)

QUOTED_PRODUCT_NAME_RULE = (
    "For subject/item quoted names, omit external generic labels (товар/услуга) and outer quotation "
    "delimiters; preserve text and embedded quotes inside the name. Keep labels inside the name. "
    "Evidence stays verbatim."
)

LINE_CHARGES_RULE = (
    "For goods, 'по ... рублей каждый' and 'за килограмм/литр/метр' give unit prices, not line "
    "totals. Without a stated invoice total, sum only when every active line has an explicit "
    "whole-line fee or known quantity*unit_price. Unknown quantities mean amount missing/null, not "
    "a partial sum; unclear price scope means ambiguous/null. Never infer quantity=1 from line "
    "count. Explicit batch/work fees override unit-price interpretation and may be summed without "
    "quantity or inventing unit prices (shared fee for two lines: fee + fee). Example: "
    "apples/pears at 100 each, apples=2, pears unknown: total missing; pears=3 gives 500. Entire "
    "apple batch=100 and pear batch=150 gives 250; two entire batches at 100 gives 200."
)

INVOICE_AMOUNT_RULE = (
    "amount_rub is the invoice total. A single unqualified amount attached to an invoice request "
    "is its total unless context says unit price, partial charge or alternatives: 'Выставь счёт за "
    "стулья, 5000 рублей' and '...на 5000 рублей' both mean total=5000. An explicit total "
    "(всего/итого/на сумму) stays known without quantity; infer no item attributes. Unit-price "
    "cues take precedence: 'стулья по 5000 рублей' leaves total missing without quantity. Outside "
    "this convention, unresolved unit-price versus line-total wording means ambiguous/null with "
    "evidence. Distinguish a service's whole fee from per-hour/session prices. "
    + LINE_CHARGES_RULE + " A VAT percentage alone does not authorize adding tax."
)

INVOICE_SUBJECT_RULE = (
    "Subject and items are not mutually exclusive: one billed category supports both without "
    "quantity, price or a separate purpose sentence. For different goods require a stated common "
    "purpose; never concatenate names."
)

ITEM_UNIT_RULE = (
    "Units (e.g. 2 комплекта): copy this line's latest stated form (шт. stays шт.). Prefer the quantity's unit: "
    "'4 часа по 2400 рублей за час' -> часа. Use the price's unit only if no quantity unit is stated. "
    "No separate unit label is needed. A product name is not a unit ('2 консультации' -> null). No unit -> null."
)

ITEM_QUANTITY_RULE = (
    "quantity is the stated or unambiguously calculated count for this line only. Never infer "
    "quantity=1 from a name, fee or total; the number of named lines is not quantity. If "
    "unsupported, null is correct."
)

ITEM_UNIT_PRICE_RULE = (
    "unit_price_rub is one unit's price for this line. A stated price ('по ... рублей каждый') "
    "remains known with unknown quantity; never infer quantity=1. An entire batch or work fee "
    "alone is not a unit price. A same-line total and positive count support total/count; assume "
    "one unit price unless different prices are stated. Preserve differently priced lines; never "
    "average across them or allocate a shared total. Otherwise null is correct."
)


def equivalent_field_values(document_type, name, left, right):
    """Только точное равенство или одна явно разрешённая группа вариантов."""
    if left == right:
        return True
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    return any(left in group and right in group
               for group in PERMITTED_ALIASES.get((document_type, name), ()))
