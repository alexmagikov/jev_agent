"""Единый контракт области ремонта для LLM и пайплайна."""

import re

from src.cases.document_type import DocumentMessage
from src.extraction import _NUMBERS
from src.schemas.extraction_schemas import ExtractionError, ExtractionProposal, SourceQuote, SourceFragmentSelection


_ATTRIBUTE_PATH = re.compile(r"items\[(0|[1-9][0-9]*)\]\.(description|quantity|unit|unit_price_rub)")
_PLANS = {"quantity": "quantity_calculation", "unit_price_rub": "unit_price_calculation"}


def _source_parts(base: ExtractionProposal) -> dict[str, tuple[SourceQuote, str, tuple]]:
    """Разрешённые части источников с проверкой индексов и имён операндов."""
    result = {}
    for name, field in base.fields.items():
        for index, source in enumerate(field.evidence):
            for part in ("message_id", "quote"):
                result[f"{name}.evidence[{index}].{part}"] = (source, part, (len(field.evidence),))

        plans = [(f"{name}.calculation", field.calculation)]
        if name == "items" and isinstance(field.value, list):
            for index, item in enumerate(field.value):
                plans.extend((f"items[{index}].{plan_name}", getattr(item, plan_name))
                             for plan_name in ("quantity_calculation", "unit_price_calculation"))
                
        for prefix, plan in plans:
            if plan is not None:
                for index, operand in enumerate(plan.operands):
                    for part in ("message_id", "quote"):
                        result[f"{prefix}.operands[{index}].source.{part}"] = (
                            operand.source, part, (len(plan.operands), operand.name),
                        )
    return result


def source_repair_context(
    base: ExtractionProposal, paths: list[str], messages: list[DocumentMessage],
) -> list[dict[str, str]]:
    """Исходный текст для недословных цитат в разрешённой области ремонта.

    Использует те же пути, что и merge_repair. Контекст идёт только в запрос LLM,
    а не в сообщение исключения; цитаты не исправляются автоматически.
    """
    sources = _source_parts(base)
    indexed = {message.id: message for message in messages}
    result = []
    for path in paths:
        entry = sources.get(path)
        if entry is None:
            continue
        source, part, _ = entry
        message = indexed.get(source.message_id)
        if part != "quote" or message is None or message.role != "user":
            continue
        if source.quote.strip() and source.quote in message.text:
            continue
        result.append({"path": path, "message_id": source.message_id,
                       "invalid_quote": source.quote, "source_text": message.text})
    return result


def quote_fragment_options(context: list[dict[str, str]]) -> list[dict]:
    """Готовые дословные фрагменты; операндам предлагаются только числовые токены."""
    result = []
    for index, source in enumerate(context):
        text = source["source_text"]
        if ".operands[" in source["path"]:
            fragments = [match.group() for match in _NUMBERS.finditer(text)]
        else:
            fragments = [part.strip() for part in re.split(r"(?<=[.!?;])\s+|[\r\n]+", text) if part.strip()]
            # Полная реплика остаётся вариантом для связи сведений из разных предложений.
            fragments.append(text)
        unique = list(dict.fromkeys(fragments))
        result.append({"path": source["path"], "message_id": source["message_id"], "fragments": [
            {"fragment_id": f"q{index}f{number}", "text": fragment}
            for number, fragment in enumerate(unique)
        ]})
    return result


def apply_quote_fragment_selections(
    base: ExtractionProposal, options: list[dict], selections: list[SourceFragmentSelection],
) -> ExtractionProposal:
    """Копирует выбранные фрагменты, не принимая от LLM новых текстов или значений."""
    by_path = {entry["path"]: entry for entry in options}
    if len(selections) != len(by_path) or {choice.path for choice in selections} != set(by_path):
        raise ExtractionError("Ремонт цитат вернул неполный или неожиданный набор путей", code="invalid_source")
    result = base.model_copy(deep=True)
    sources = _source_parts(result)
    for choice in selections:
        entry = by_path[choice.path]
        fragments = {fragment["fragment_id"]: fragment["text"] for fragment in entry["fragments"]}
        if choice.fragment_id is None or choice.fragment_id not in fragments:
            raise ExtractionError("Ремонт цитаты не выбрал допустимый фрагмент", code="invalid_source")
        source, part, _ = sources[choice.path]
        if part != "quote" or source.message_id != entry["message_id"]:
            raise ExtractionError("Ремонт цитаты не может менять её источник", code="invalid_source")
        source.quote = fragments[choice.fragment_id]
    return result


def _validate_paths(base: ExtractionProposal, paths: list[str]) -> set[str]:
    if not paths:
        raise ValueError("Нужна непустая область ремонта")
    result = set(paths)
    sources = _source_parts(base)
    evidence_paths = {f"{name}.evidence" for name in base.fields}
    for path in result:
        if path in base.fields or path in sources or path in evidence_paths:
            continue
        match = _ATTRIBUTE_PATH.fullmatch(path)
        items = base.fields.get("items")
        if (match is None or items is None or items.status != "fillable"
                or not isinstance(items.value, list) or int(match[1]) >= len(items.value)):
            raise ValueError("Неизвестный путь ремонта: " + path)
    return {path for path in result if not any(
        parent != path and (path.startswith(parent + ".") or path.startswith(parent + "["))
        for parent in result
    )}


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
    return sorted({path.split("[", 1)[0].split(".", 1)[0] for path in paths})


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

    old_sources, new_sources = _source_parts(base), _source_parts(candidate)
    result_sources = _source_parts(result)
    for path in allowed:
        if path.endswith(".evidence"):
            name = path.split(".", 1)[0]
            result.fields[name].evidence = [source.model_copy(deep=True) for source in candidate.fields[name].evidence]
        elif path in old_sources:
            if path not in new_sources or old_sources[path][2] != new_sources[path][2]:
                raise ExtractionError("Точечный ремонт источника не может менять число цитат или порядок операндов")
            source, part, _ = new_sources[path]
            setattr(result_sources[path][0], part, getattr(source, part))
    attribute_paths = [path for path in allowed if _ATTRIBUTE_PATH.fullmatch(path)]
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
    seen = {(quote.message_id, quote.quote) for quote in result.fields["items"].evidence}
    for quote in new.evidence:
        if (quote.message_id, quote.quote) not in seen:
            result.fields["items"].evidence.append(quote.model_copy(deep=True))
            seen.add((quote.message_id, quote.quote))
    return result
