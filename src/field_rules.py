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


def equivalent_field_values(document_type, name, left, right):
    """Только точное равенство или одна явно разрешённая группа вариантов."""
    if left == right:
        return True
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    return any(left in group and right in group
               for group in PERMITTED_ALIASES.get((document_type, name), ()))
