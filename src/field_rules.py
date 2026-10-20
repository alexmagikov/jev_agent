"""Общий контракт извлечения, смысловой проверки и разрешённых вариантов значений.

Здесь нет эталонов или номеров кейсов. Строгая JSON-метрика остаётся отдельной:
эквивалентность ограничена явно разрешёнными вариантами двух полей перевода.
"""

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

ITEM_DESCRIPTION_RULE = (
    "Normalize an ordinary item description to singular nominative when the conversion "
    "is unambiguous, even when quantity is greater than one or unknown. Keep the actual "
    "quantity in quantity; grammatical singular does not mean quantity=1. Preserve "
    "characteristics, composition and meaning. Do not change names in quotation marks, "
    "established product names, plural-only nouns (such as ножницы), or wording whose "
    "singular form changes the meaning or is ambiguous. For a quoted item name, preserve "
    "the text inside exactly, but omit outer quotation delimiters used only to mark that "
    "name in the source. Do not remove quotation marks embedded within the actual name. "
    "This normalization applies only "
    "to items.value[i].description, not to subject, company names or evidence quotes."
)

ENUMERATED_ITEM_CHARGES_RULE = (
    "Task convention for an explicit finite list of named billed goods: wording such as "
    "'товары А и Б, по ... рублей каждый' assigns that charge to each named bill line. "
    "Sum the charge once for each listed line even with unknown quantities. This is a "
    "calculable invoice total, not missing or ambiguous merely because quantity is null. "
    "Do not infer quantity=1 or any unstated measurement unit. A common quoted price may "
    "be reused in the expression (fee + fee for two lines), without inventing a numeric "
    "operand for the number of lines. This convention does not apply to explicit prices "
    "per measurement unit (за килограмм, за литр, за метр) with unknown quantities, open "
    "lists, alternative rather than jointly billed goods, or unresolved charge allocation. "
    "When actual quantities are explicitly supplied, respect them and the stated price "
    "role; do not replace quantity-times-unit-price with one charge per line."
)

INVOICE_AMOUNT_RULE = (
    "amount_rub is the invoice total, not any number accompanied by рублей. "
    "Resolve the numerical role from the entire dialogue. A bare countable-goods name "
    "followed by a price, without a quantity or a total/unit-price cue, leaves unit price "
    "versus line total unresolved: use status=ambiguous, value=null and cite that wording. "
    "An explicit invoice total (счёт на ... рублей, услуга на ... рублей, на сумму, всего, "
    "итого, общей стоимостью) supports the total; "
    "a stated unit price (по, за единицу) and known quantity support a calculated total. "
    "Do not mark an explicit total ambiguous just because quantity is missing. "
    "A stated fee for named work/service can be the total fee for that work without a "
    "quantity; this does not imply quantity=1 or a unit price. Distinguish a work fee "
    "from a per-session/per-hour price using the user's wording. "
    + ENUMERATED_ITEM_CHARGES_RULE + " "
    "A VAT percentage alone does not authorize adding tax."
)

INVOICE_SUBJECT_RULE = (
    "A single explicitly named billed goods/service category supports subject as well as "
    "the item's description. Missing quantity or an ambiguous price does not erase that "
    "known subject. Do not require a separate purpose sentence for a single named category. "
    "For several different named goods, do not invent an umbrella purpose by concatenating them."
)

ITEM_UNIT_RULE = (
    "unit is an explicitly supplied measurement or packaging unit, such as штук, кг, часов "
    "or коробок. A counted goods/service noun is not itself a measurement unit: in a "
    "quantity plus service-name phrase, keep the known quantity and description but "
    "leave unit=null unless a separate unit is supplied. Do not manufacture a unit from "
    "the description."
)

ITEM_QUANTITY_RULE = (
    "quantity describes this single line item only. Use an explicitly supplied count "
    "or an unambiguous calculation from source numbers that belong to this item. "
    "Do not invent quantity=1 from the existence of an item, its name, a work fee, "
    "or a total alone. The number of named bill lines is not the physical quantity "
    "of each line. If the source does not determine the quantity, null is correct."
)

ITEM_UNIT_PRICE_RULE = (
    "unit_price_rub describes the price of one unit of this single line item. "
    "An explicitly stated unit price remains known even with unknown quantity; "
    "quantity=null alone is not a reason to discard that price. A price stated "
    "for each explicitly named good ('по ... рублей каждый') supports its unit price "
    "without inferring quantity=1. A work fee or line total alone does not establish "
    "a unit price. A line total and positive quantity for this same line support "
    "total/count unless different unit prices within the line are indicated. "
    "Do not use a combined total/count across different goods or invent an allocation. "
    "If no direct unit price or unambiguous same-line calculation is supported, null is correct."
)


def equivalent_field_values(document_type, name, left, right):
    """Только точное равенство или одна явно разрешённая группа вариантов."""
    if left == right:
        return True
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    return any(left in group and right in group
               for group in PERMITTED_ALIASES.get((document_type, name), ()))
