"""Единый контракт области ремонта для LLM и пайплайна."""

import re

from src.schemas.extraction_schemas import ExtractionError, ExtractionProposal


_ATTRIBUTE_PATH = re.compile(r"items\[(0|[1-9][0-9]*)\]\.(description|quantity|unit|unit_price_rub)")
_PLANS = {"quantity": "quantity_calculation", "unit_price_rub": "unit_price_calculation"}


def _validate_paths(base: ExtractionProposal, paths: list[str]) -> set[str]:
    if not paths:
        raise ValueError("Нужна непустая область ремонта")
    result = set(paths)
    for path in result:
        if path in base.fields:
            continue
        match = _ATTRIBUTE_PATH.fullmatch(path)
        items = base.fields.get("items")
        if (match is None or items is None or items.status != "fillable"
                or not isinstance(items.value, list) or int(match[1]) >= len(items.value)):
            raise ValueError("Неизвестный путь ремонта: " + path)
    if "items" in result:
        result = {path for path in result if not path.startswith("items[")}
    return result


def plan_repair(base: ExtractionProposal, paths: list[str]) -> list[str]:
    """Добавляет только существующие расчётные зависимости, не выдумывая значения.

    Планы используют исходные цитаты, а не ссылки между полями. Поэтому для
    числового ремонта консервативно перепроверяем расчёты этой же позиции и
    вычисленную сумму, а для суммы — расчётные атрибуты всех позиций.
    Явно указанная сумма не становится ремонтируемой автоматически.
    """
    result = _validate_paths(base, paths)
    original = result.copy()
    items = base.fields.get("items")
    numeric_items_changed = "items" in result
    if items is not None and isinstance(items.value, list) and "items" not in result:
        for index, item in enumerate(items.value):
            prefix = f"items[{index}]."
            if "amount_rub" in result or any(prefix + attr in result for attr in ("quantity", "unit", "unit_price_rub")):
                numeric_items_changed = True
                for attribute, plan in _PLANS.items():
                    if getattr(item, plan) is not None:
                        result.add(prefix + attribute)
    amount = base.fields.get("amount_rub")
    if numeric_items_changed and amount is not None and amount.calculation is not None:
        result.add("amount_rub")
    # Замыкание должно быть идемпотентным: extract() и pipeline используют один план.
    if result != original:
        return plan_repair(base, sorted(result))
    return sorted(result)


def repair_roots(paths: list[str]) -> list[str]:
    return sorted({path.split("[", 1)[0] for path in paths})


def merge_repair(
    base: ExtractionProposal, candidate: ExtractionProposal, paths: list[str],
) -> ExtractionProposal:
    """Меняет только разрешённые пути; не мутирует кандидатов и не расширяет план.

    Для атрибутов LLM возвращает поле items целиком, но Python сохраняет все
    остальные значения. Несовместимая форма или смена идентичности — отказ,
    а не применение чисел к потенциально чужой позиции.
    """
    allowed = _validate_paths(base, paths)
    missing = set(repair_roots(list(allowed))) - set(candidate.fields)
    if missing:
        raise ExtractionError("В ответе ремонта отсутствуют запрошенные поля")
    result = base.model_copy(deep=True)
    for path in allowed:
        if path in base.fields:
            result.fields[path] = candidate.fields[path].model_copy(deep=True)
    attribute_paths = [path for path in allowed if path.startswith("items[")]
    if not attribute_paths:
        return result
    old, new = base.fields["items"], candidate.fields["items"]
    if (new.status != old.status or not isinstance(new.value, list)
            or len(new.value) != len(old.value)):
        raise ExtractionError("Точечный ремонт не может менять статус или число позиций")
    for index, (before, after) in enumerate(zip(old.value, new.value, strict=True)):
        description_path = f"items[{index}].description"
        if before.description != after.description:
            if description_path not in allowed or any(
                other.description == after.description and other.description != before.description
                for other in old.value
            ):
                raise ExtractionError("Точечный ремонт не может переставлять или заменять другие позиции")
    for path in attribute_paths:
        match = _ATTRIBUTE_PATH.fullmatch(path)
        index, attribute = int(match[1]), match[2]
        target = result.fields["items"].value[index]
        source = new.value[index]
        setattr(target, attribute, getattr(source, attribute))
        if attribute in _PLANS:
            plan = getattr(source, _PLANS[attribute])
            setattr(target, _PLANS[attribute], plan.model_copy(deep=True) if plan is not None else None)
    # Evidence принадлежит всему полю: сохраняем опору неизменённых атрибутов.
    seen = {(quote.message_id, quote.quote) for quote in old.evidence}
    for quote in new.evidence:
        if (quote.message_id, quote.quote) not in seen:
            result.fields["items"].evidence.append(quote.model_copy(deep=True))
            seen.add((quote.message_id, quote.quote))
    return result
