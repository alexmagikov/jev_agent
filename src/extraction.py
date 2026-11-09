"""Предложения LLM, проверка источников и арифметика без выполнения кода модели."""

import ast
import json
import math
import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, DecimalException, localcontext
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, TypeAdapter, ValidationError

from src.cases.document_type import DocumentMessage
from src.schemas.document_schemas import (
    DocumentValidation,
    ValidationIssue,
    FIELDS_BY_TYPE,
    field_root,
)
from src.schemas.extraction_schemas import (
    Calculation,
    ProposedField,
    ExtractionProposal,
    PreparedFields,
    SourceQuote,
    SupportedDocumentType,
    ExtractionError,
    InvoiceExtractionProposal,
    TransferExtractionProposal
)


PROPOSAL_MODELS = {
    "invoice_draft": InvoiceExtractionProposal,
    "transfer_draft": TransferExtractionProposal
}


def get_proposal_model(document_type: SupportedDocumentType) -> type[BaseModel]:
    """Типизированный ответ LLM: набор полей и типы значений из итогового шаблона.

    Отдельная модель нужна только на границе с LLM. Дальше пайплайн по-прежнему
    работает с ExtractionProposal и словарём fields; формат результата не меняется.
    """

    return PROPOSAL_MODELS[document_type]


def typed_proposal(proposal: BaseModel, document_type: SupportedDocumentType) -> ExtractionProposal:
    """Проверяет поля по типизированной модели документа и возвращает общий ExtractionProposal."""
    checked = get_proposal_model(document_type).model_validate(proposal.model_dump())
    return ExtractionProposal.model_validate(checked.model_dump())


def get_proposal_schema(document_type: SupportedDocumentType, *, field_names: list[str] | None = None) -> dict:
    """Создаёт JSON-схему ответа, который должна вернуть LLM. Схема описывает допустимые поля, их типы и обязательность."""
    schema = get_proposal_model(document_type).model_json_schema()
    if field_names is not None:
        # При ремонте сервер должен генерировать только отмеченные поля.
        # model_json_schema() возвращает новый словарь: полная схема не изменяется.
        fields_schema = schema["$defs"][schema["properties"]["fields"]["$ref"].split("/")[-1]]
        if not field_names or not set(field_names) <= set(fields_schema["properties"]):
            raise ValueError("Для ремонта нужны существующие поля текущего шаблона")
        fields_schema["properties"] = {name: fields_schema["properties"][name] for name in field_names}

    def require_properties(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object":
                node["required"] = list(node["properties"])
            for value in node.values():
                require_properties(value)
        elif isinstance(node, list):
            for value in node:
                require_properties(value)

    require_properties(schema)
    return schema


@lru_cache(maxsize=2)
def get_template(document_type: SupportedDocumentType) -> dict:
    """Загружает шаблон документа; кеширует результат отдельно для счёта и перевода."""
    path = Path(__file__).resolve().parents[1] / "data" / "form_templates.json"
    return json.loads(path.read_text(encoding="utf-8"))[document_type]


def validate_messages(messages: list[DocumentMessage]) -> None:
    """Проверяет, что диалог непустой и ID его реплик уникальны."""
    if not messages or len({message.id for message in messages}) != len(messages):
        raise ExtractionError("Нужны непустой диалог и уникальные ID реплик")


def message_index(messages: list[DocumentMessage]) -> dict[str, DocumentMessage]:
    """Индекс реплик по ID; уникальность ID проверяет validate_messages на входе пайплайна."""
    return {message.id: message for message in messages}


def _check_source(source: SourceQuote, indexed: dict[str, DocumentMessage], *, path: str) -> None:
    """Путь состоит из имён полей схемы и индексов, без цитат или модельных ID в ошибке."""
    message = indexed.get(source.message_id)

    if message is None or message.role != "user":
        raise ExtractionError(
            f"{path}.message_id: Источник должен ссылаться на существующую реплику user",
            code="invalid_source", repair_paths=(f"{path}.message_id", f"{path}.quote"),
        )

    if not source.quote.strip() or source.quote not in message.text:
        raise ExtractionError(
            f"{path}.quote: Цитата источника отсутствует в указанной реплике. "
            "Скопируй дословный фрагмент исходного messages[message_id].text; "
            "не меняй кавычки, пробелы и пунктуацию.",
            code="invalid_source", repair_paths=(f"{path}.quote",),
        )


def _has_whole_numeric_source(
    value: str,
    sources: list[SourceQuote],
    indexed: dict[str, DocumentMessage],
) -> bool:
    """Номер целиком входит в цитату и не обрезан относительно исходного сообщения."""
    if not value:
        return False

    pattern = re.compile(rf"(?<!\d){re.escape(value)}(?!\d)")
    for source in sources:
        text = indexed[source.message_id].text
        for quote_match in re.finditer(re.escape(source.quote), text):
            # Не ограничиваем поиск концом цитаты: lookahead должен видеть
            # следующую цифру в исходном тексте, даже если цитата обрезана.
            value_match = pattern.search(text, quote_match.start())
            if value_match is not None and value_match.end() <= quote_match.end():
                return True
    return False


_EMAIL_PARTS = re.compile(r'''[^\s<>()\[\]{}«»“”‘’",;:!?]+''')
_CARD_MASK_CHARS = r"*xXхХ•●·∙×_"
_CARD_SEPARATOR = r"[\s\-‐‑–]+"
_CARD_MASK_GROUP = rf"[0-9]*[{_CARD_MASK_CHARS}][0-9{_CARD_MASK_CHARS}]*"
_CARD_PARTS = re.compile(
    rf"(?<![\w{_CARD_MASK_CHARS}])(?:"
    rf"(?:[0-9]+{_CARD_SEPARATOR})*"
    rf"(?:{_CARD_MASK_GROUP}{_CARD_SEPARATOR})*{_CARD_MASK_GROUP}"
    # После слитной маски ****9090 следующее число уже не часть карты. После отдельной
    # скрытой группы идут открытые: «2202 **** 1234 9090» — одна маска. Следующие группы
    # ровно по 4 цифры и не больше двух, чтобы не захватить сумму через запятую или длинное число.
    rf"(?:(?<=[{_CARD_MASK_CHARS}]){_CARD_SEPARATOR}[0-9]+(?:{_CARD_SEPARATOR}[0-9]{{4}}){{0,2}})?"
    rf"|[0-9]+(?:{_CARD_SEPARATOR}[0-9]+)*)(?![\w{_CARD_MASK_CHARS}])"
)


def _has_whole_contact_source(
    name: str, value: str, sources: list[SourceQuote], indexed: dict[str, DocumentMessage],
) -> bool:
    """Целый email/шаблон карты в цитате; границы берутся из исходной реплики.

    Это проверка копирования, а не принадлежности или актуальности реквизита.
    Формат email проверяется отдельно, поэтому исходный ошибочный адрес не теряется.
    """
    is_email = name == "recipient_email"
    for source in sources:
        text = indexed[source.message_id].text
        spans = []
        for match in (_EMAIL_PARTS if is_email else _CARD_PARTS).finditer(text):
            start, end = match.span()
            if is_email:
                # Апостроф внутри адреса сохраняется; парные внешние — кавычки.
                if text[start:end].startswith("'") and text[start:end].endswith("'") and end - start > 1:
                    start, end = start + 1, end - 1
                spans.append((start, end))
                # Точка после адреса может завершать предложение.
                end = start + len(text[start:end].rstrip("."))
            if end > start:
                spans.append((start, end))
        for quote in re.finditer(re.escape(source.quote), text):
            contained = [(start, end) for start, end in spans
                         if quote.start() <= start and end <= quote.end()]
            if any(text[start:end] == value for start, end in contained):
                return True
            if is_email and invalid_format(name, value):
                # Некорректный адрес может содержать пробелы. Разрешаем его целиком,
                # но не отрезанный префикс/суффикс исходного адреса.
                starts = {start for start, _ in contained}
                ends = {end for _, end in contained}
                for match in re.finditer(re.escape(value), text):
                    if match.start() in starts and match.end() in ends:
                        return True
    return False


def _has_whole_unit_source(
    unit: str, sources: list[SourceQuote], indexed: dict[str, DocumentMessage],
) -> bool:
    """Точная форма в указанных репликах, не часть слова/единицы; перед ней допустимо число.

    Проверка только лексическая: актуальность и принадлежность позиции оценивает Jev.
    У сокращений вроде «шт.» точка сохраняется как часть обозначения.
    После символов вроде «кг» и полных слов точка может завершать предложение.
    """
    if not unit.strip():
        return False
    # Слева запрещены буквы/подчёркивание, но разрешена цифра (2м, 2,5кг).
    # Справа запрещены любые word-символы, включая цифры/степени (м != м2/м²).
    dotted_abbreviations = {"шт", "компл", "уп", "упак", "кор", "пач", "рул", "наб", "ед", "экз"}
    ending = r"(?![\w.])" if unit.split()[-1].casefold() in dotted_abbreviations else r"(?!\w)"
    pattern = re.compile(rf"(?<![^\W\d]){re.escape(unit)}{ending}")
    return any(pattern.search(indexed[source.message_id].text) for source in sources)


_NUMBER = r"[+-]?(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?"
# Число может примыкать к известной единице, но не к произвольному слову/артикулу.
_ATTACHED_UNIT = (
    r"(?:руб(?:лей|ля|ль)?|р|коп(?:еек|ейки|ейку|ейка)?|шт(?:ук(?:а|и|у)?)?|"
    r"кг|гр|мг|г|т|мл|л|км|см|мм|м[²³23]?|ч(?:ас(?:а|ов)?)?|мин|сек|"
    r"уп(?:ак(?:ов(?:ка|ки|ок))?)?|короб(?:ка|ки|ок)|комплект(?:а|ов)?)"
)
NUMBERS = re.compile(
    rf"(?<!\w)(?<!\d[.,]){_NUMBER}"
    # При неудачном полном совпадении нельзя принять начало дроби или группы тысяч.
    rf"(?:(?!\w|[.,]\d|[ \u00a0\u202f]\d{{3}}(?!\d))|(?={_ATTACHED_UNIT}(?!\w)))",
    re.IGNORECASE,
)
_RUB_KOP = re.compile(
    rf"({_NUMBER})\s*(?:рублей|рубля|рубль|руб\.?)\s*"
    r"(\d{1,2})\s*(?:копеек|копейки|копейку|копейка|коп\.?)", re.IGNORECASE,
)


def _decimal(text: str) -> Decimal:
    if len(text) > 120 or re.fullmatch(_NUMBER, text.strip()) is None:
        raise ExtractionError("Операнд должен быть точным числовым фрагментом, без единиц", code="invalid_operand")
    value = Decimal(re.sub(r"[\s\u00a0\u202f]", "", text).replace(",", "."))
    if not value.is_finite() or abs(value) > Decimal("1e30"):
        raise ExtractionError("Число выходит за допустимый диапазон", code="invalid_operand")
    return value


def _numbers_in(quotes: list[str]) -> set[Decimal]:
    numbers: set[Decimal] = set()
    for quote in quotes:
        for match in NUMBERS.finditer(quote):
            try:
                numbers.add(_decimal(match.group()))
            except ExtractionError:
                continue
        for match in _RUB_KOP.finditer(quote):
            rubles = _decimal(match[1])
            amount = abs(rubles) + Decimal(match[2]) / 100
            numbers.add(-amount if rubles.is_signed() else amount)
    return numbers


def _calculate(
    plan: Calculation, indexed: dict[str, DocumentMessage], target: str, *, source_path: str,
) -> tuple[float, str]:
    values: dict[str, Decimal] = {}
    for index, operand in enumerate(plan.operands):
        _check_source(operand.source, indexed, path=f"{source_path}.operands[{index}].source")

        if not any(
            match.group() == operand.source.quote.strip()
            for match in NUMBERS.finditer(indexed[operand.source.message_id].text)
        ):
            if re.fullmatch(_NUMBER, operand.source.quote.strip()):
                raise ExtractionError("Операнд должен быть целым числовым фрагментом, а не частью числа", code="invalid_operand")
            # Текст уходит в ремонт: фраза вместо числа требует иной подсказки, чем часть числа.
            raise ExtractionError(
                "quote операнда — ровно одно число без слов и единиц ('8700', а не '5 шт. по 8700 рублей'). "
                "Для итога по строкам дай каждому количеству и цене свой операнд: q1 * p1 + q2 * p2.",
                code="invalid_operand",
            )
        if operand.name in values:
            raise ExtractionError("Повторяющееся имя операнда", code="invalid_calculation")
        values[operand.name] = _decimal(operand.source.quote)
    try:
        tree = ast.parse(plan.expression, mode="eval")
        if len(list(ast.walk(tree))) > 64:
            raise ExtractionError("Слишком сложная формула", code="invalid_calculation")
        used: set[str] = set()

        def evaluate(node: ast.AST) -> Decimal:
            if isinstance(node, ast.Name) and node.id in values:
                used.add(node.id)
                return values[node.id]
            # Константы — только для процентов и перевода единиц (кг↔г, ч↔мин).
            if isinstance(node, ast.Constant) and type(node.value) is int and node.value in (0, 1, 60, 100, 1000):
                return Decimal(node.value)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = evaluate(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp):
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
            if isinstance(node, ast.Constant):
                # Текст уходит в ремонт: число из реплики должно стать операндом, иначе его не проверить.
                raise ExtractionError(
                    "В expression есть число, которое не является разрешённой константой (0, 1, 60, 100, 1000). "
                    "Процент скидки и другие числа из реплик передай отдельным операндом с цитатой "
                    "(например discount с quote '5') и используй его имя: q * p * (100 - discount) / 100.",
                    code="invalid_calculation",
                )
            raise ExtractionError("В формуле разрешены только операнды, +, -, *, / и скобки", code="invalid_calculation")

        # Сумма и цена единицы — деньги: округляем до копеек, половину копейки вверх.
        money = target == "amount_rub" or target.endswith(".unit_price_rub")
        with localcontext() as context:
            context.prec = 40
            value = evaluate(tree.body)
            rounded = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) if money else value
        if used != set(values):
            raise ExtractionError("Все объявленные операнды должны использоваться в формуле", code="invalid_calculation")
        numeric = float(rounded)
        if not math.isfinite(numeric) or Decimal(str(numeric)) != rounded:
            raise ExtractionError("Результат нельзя точно представить в JSON; правило округления не задано", code="invalid_calculation")
    except (SyntaxError, DecimalException, OverflowError, RecursionError) as error:
        raise ExtractionError("Некорректная арифметическая формула", code="invalid_calculation") from error
    operands = ", ".join(f"{name}={value}" for name, value in values.items())
    derivation = f"{target} = {plan.expression}; {operands}; result={value}"
    if rounded != value:
        derivation += f"; rounded_to_kopecks={rounded}"
    return numeric, derivation


_REQUISITE_FORMATS = {
    "recipient_inn": (r"(?:[0-9]{10}|[0-9]{12})", "Exactly 10 or 12 ASCII digits."),
    "recipient_account": (r"[0-9]{20}", "Exactly 20 ASCII digits."),
    "recipient_bik": (r"[0-9]{9}", "Exactly 9 ASCII digits."),
    "recipient_email": (r"[^\s@]+@[^\s@]+\.[^\s@]+", "Nonempty local@domain.suffix, no whitespace or extra @."),
}


def invalid_format(name: str, value: object) -> bool:
    """Значение не соответствует формату поля: реквизиты, дата, сумма, ставка НДС."""

    if name in _REQUISITE_FORMATS:
        pattern, _ = _REQUISITE_FORMATS[name]
        return not isinstance(value, str) or re.fullmatch(pattern, value) is None

    if name == "basis_date":
        try:
            return not isinstance(value, str) or date.fromisoformat(value).isoformat() != value
        except ValueError:
            return True

    if name == "amount_rub":
        return not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0

    if name == "vat_rate_percent":
        return not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 100
    return False


def requisite_format_context(name: str, value: object = None) -> dict | None:
    """Факт проверки формы исходного значения, не вердикт о его смысле/актуальности.

    При null остаётся только правило: по цитате не угадываем, какое число
    относится к полю, и не утверждаем, что Python уже проверил invalid/null.
    """
    if name not in _REQUISITE_FORMATS:
        return None

    context = {"rule": _REQUISITE_FORMATS[name][1]}
    if value is not None:
        context.update(checked_by="python", input_value=value,
                       matches_format=not invalid_format(name, value))
        if isinstance(value, str):
            context["input_length"] = len(value)
    return context


def requisite_source_candidates(name: str, messages: list[DocumentMessage]) -> list[dict] | None:
    """Ищет в сообщениях пользователя строки, подходящие по формату на заданный реквизит"""

    if name not in _REQUISITE_FORMATS:
        return None

    pattern = _REQUISITE_FORMATS[name][0]
    if name != "recipient_email":
        pattern = rf"(?<!\d){pattern}(?!\d)"

    candidates = []
    seen = set()
    for message in messages:
        if message.role != "user":
            continue

        for match in re.finditer(pattern, message.text):
            value = match.group()
            identity = (message.id, value)
            if identity not in seen and not invalid_format(name, value):
                candidates.append({"message_id": message.id, "quote": value})
                seen.add(identity)
    return candidates


def _proposal_sources(proposal: ExtractionProposal) -> list[tuple[SourceQuote, str]]:
    """Собирает цитаты поля и операндов вместе с точными путями ошибок."""
    sources = []
    for name, field in proposal.fields.items():
        sources.extend((source, f"{name}.evidence[{index}]") for index, source in enumerate(field.evidence))
        plans = [(f"{name}.calculation", field.calculation)]
        if name == "items" and isinstance(field.value, list):
            for index, item in enumerate(field.value):
                plans.extend((f"items[{index}].{plan_name}", getattr(item, plan_name))
                             for plan_name in ("quantity_calculation", "unit_price_calculation"))
        for path, plan in plans:
            if plan is not None:
                sources.extend((operand.source, f"{path}.operands[{index}].source")
                               for index, operand in enumerate(plan.operands))
    return sources


def _raw_field(field: ProposedField) -> dict:
    """Переносит исходные значения в черновик до проверки и выполнения формул."""
    return {
        "status": field.status,
        "value": [
            item.model_dump(exclude={"quantity_calculation", "unit_price_calculation"})
            for item in field.value
        ] if isinstance(field.value, list) else field.value,
        "evidence": list(dict.fromkeys(source.message_id for source in field.evidence)),
        "derivation": None,
    }


# Реквизиты, значение которых должно дословно входить в цитату.
_VERBATIM_FIELDS = frozenset({
    "recipient_inn", "recipient_email", "recipient_account", "recipient_bik",
    "destination_card_masked", "basis_number",
})


def _apply_calculation(
    plan: Calculation, target: str, checked_field: dict, indexed: dict[str, DocumentMessage],
    calculated: set[str], *, source_path: str,
) -> float:
    """Выполняет одну формулу и дописывает её запись и источники в checked_field."""
    if len(calculated) >= 32:
        raise ExtractionError("Допустимо не больше 32 вычислений на предложение", repair_paths=(target,))

    try:
        value, formula = _calculate(plan, indexed, target, source_path=source_path)
    except ExtractionError as error:
        if not error.repair_paths:
            error.repair_paths = (target,)
        raise
    checked_field["derivation"] = "; ".join(filter(None, [checked_field["derivation"], formula]))
    checked_field["evidence"] = list(dict.fromkeys([
        *checked_field["evidence"], *(operand.source.message_id for operand in plan.operands),
    ]))
    calculated.add(target)
    return value


def _calculate_field(
    name: str, field: ProposedField, checked_field: dict, indexed: dict[str, DocumentMessage], calculated: set[str],
) -> None:
    """Выполняет формулы поля: общую сумму и количество или цену позиций счёта."""
    if field.calculation is not None:
        if name != "amount_rub":
            raise ExtractionError(f"{name}: недопустимое поле для вычисления", repair_paths=(name,))
        if field.status != "fillable":
            raise ExtractionError("Нельзя вычислять незаполняемое поле", repair_paths=(name,))
        if field.value is not None:
            raise ExtractionError(f"{name}: при calculation значение должно быть null", code="conflicting_calculation_value", repair_paths=(name,))

        checked_field["value"] = _apply_calculation(
            field.calculation, name, checked_field, indexed, calculated, source_path=f"{name}.calculation",
        )

    if name == "items" and isinstance(field.value, list):
        for index, item in enumerate(field.value):
            for key, plan_name, plan in (
                ("quantity", "quantity_calculation", item.quantity_calculation),
                ("unit_price_rub", "unit_price_calculation", item.unit_price_calculation),
            ):
                if plan is None:
                    continue
                target = f"items[{index}].{key}"
                if getattr(item, key) is not None:
                    raise ExtractionError(f"{target}: при расчёте значение должно быть null", code="conflicting_calculation_value", repair_paths=(target,))
                checked_field["value"][index][key] = _apply_calculation(
                    plan, target, checked_field, indexed, calculated, source_path=f"items[{index}].{plan_name}",
                )


def _check_requisite_copy(name: str, value: object, field: ProposedField, indexed: dict[str, DocumentMessage]) -> None:
    """Реквизит скопирован из цитаты дословно и целиком, без обрезанных цифр или адреса."""
    quotes = [source.quote for source in field.evidence]
    if name in _VERBATIM_FIELDS and (not isinstance(value, str) or not any(value in q for q in quotes)):
        raise ExtractionError(f"{name}: реквизит должен дословно присутствовать в источнике", repair_paths=(name,))
    if name in {"recipient_email", "destination_card_masked"} and not _has_whole_contact_source(
        name, value, field.evidence, indexed,
    ):
        raise ExtractionError(
            f"{name}: скопируй реквизит целиком из исходного сообщения; "
            "нельзя обрезать адрес или маску в значении либо цитате. "
            "Если исходный email неверного формата, используй invalid, value=null "
            "и дословную цитату проблемного реквизита.",
            code="invalid_source", repair_paths=(name,),
        )
    if name in {"recipient_inn", "recipient_account", "recipient_bik"} and not _has_whole_numeric_source(
        value, field.evidence, indexed,
    ):
        raise ExtractionError(
            f"{name}: скопируй цифровой реквизит целиком из исходного сообщения; "
            "нельзя обрезать цифры в значении или цитате ради нужной длины. "
            "Если исходный реквизит неверного формата, используй invalid, value=null "
            "и дословную цитату проблемного реквизита.",
            code="invalid_source", repair_paths=(name,),
        )


def _cited_numbers(field: ProposedField, indexed: dict[str, DocumentMessage]) -> set[Decimal]:
    """Числа, которые есть и в цитатах поля, и в самих репликах."""
    numbers: set[Decimal] = set()
    for source in field.evidence:
        numbers.update(
            _numbers_in([source.quote]) & _numbers_in([indexed[source.message_id].text])
        )
    return numbers


def _check_items(
    value: object, field: ProposedField, indexed: dict[str, DocumentMessage],
    numbers: set[Decimal], calculated: set[str],
) -> None:
    """Позиции счёта: непустой список, описание, единица из реплик и подтверждённые числа."""
    if not isinstance(value, list) or not value:
        raise ExtractionError("items: ожидается непустой список позиций", repair_paths=("items",))

    for index, item in enumerate(value):
        if item["description"] is not None and not item["description"].strip():
            raise ExtractionError(
                f"items[{index}].description: наименование не может быть пустым; "
                "если товар или услуга не названы, верни description=null.",
                code="missing_value", repair_paths=(f"items[{index}].description",),
            )
        if item["unit"] is not None and not _has_whole_unit_source(
            item["unit"], field.evidence, indexed,
        ):
            raise ExtractionError(
                f"items[{index}].unit: единица измерения не найдена целиком в указанных репликах user. "
                "Скопируй исходную форму без изменения окончания или обозначения; "
                "если единица не указана, верни unit=null.",
                code="invalid_source", repair_paths=(f"items[{index}].unit",),
            )
        for key in ("quantity", "unit_price_rub"):
            number = item[key]
            if number is not None and f"items[{index}].{key}" not in calculated:
                if Decimal(str(number)) not in numbers:
                    raise ExtractionError(f"items[{index}].{key}: неподтверждённое число",
                                          repair_paths=(f"items[{index}].{key}",))
            if number is not None and number < 0:
                raise ExtractionError("Отрицательное количество или цена позиции", repair_paths=(f"items[{index}].{key}",))


def _prepare_field(
    name: str, field: ProposedField, indexed: dict[str, DocumentMessage], calculated: set[str],
) -> dict:
    """Проверяет одно поле; общие ограничения проверяются вызывающим методом."""
    if field.status != "fillable" and field.value is not None:
        raise ExtractionError(f"{name}: незаполняемое поле должно иметь value=null", repair_paths=(name,))
    if field.status in ("fillable", "invalid", "ambiguous") and not field.evidence:
        raise ExtractionError(f"{name}: для значения или проблемы нужны источники", code="missing_evidence",
                              repair_paths=(f"{name}.evidence",))

    checked_field = _raw_field(field)
    _calculate_field(name, field, checked_field, indexed, calculated)
    if checked_field["status"] != "fillable":
        return checked_field

    value = checked_field["value"]
    if value is None:
        raise ExtractionError(f"{name}: у заполняемого поля нет значения или расчёта", code="missing_value", repair_paths=(name,))
    if isinstance(value, str) and not value.strip():
        raise ExtractionError(f"{name}: заполняемая строка не может быть пустой или состоять из пробелов",
                              code="missing_value", repair_paths=(name,))
    _check_requisite_copy(name, value, field, indexed)

    numbers = _cited_numbers(field, indexed)
    if name in ("amount_rub", "vat_rate_percent") and name not in calculated:
        if not isinstance(value, (int, float)) or Decimal(str(value)) not in numbers:
            raise ExtractionError(f"{name}: число не подтверждено источником или формулой", code="unsupported_numeric_value", repair_paths=(name,))
    if name == "items":
        _check_items(value, field, indexed, numbers, calculated)

    if invalid_format(name, value):
        checked_field.update(status="invalid", value=None, derivation=None)
    return checked_field


def prepare_fields(
    proposal: ExtractionProposal,
    messages: list[DocumentMessage],
    document_type: SupportedDocumentType,
) -> PreparedFields:
    """Проверяет источники, вычисляет значения и отличает invalid от ошибок модели."""

    indexed = message_index(messages)
    model = FIELDS_BY_TYPE[document_type]
    if set(proposal.fields) != set(model.model_fields):
        raise ExtractionError("Предложение должно содержать ровно все поля выбранного шаблона")

    sources = _proposal_sources(proposal)

    source_errors = []
    for source, path in sources:
        try:
            _check_source(source, indexed, path=path)
        except ExtractionError as error:
            source_errors.append(error)
    if source_errors:
        raise ExtractionError(
            "\n".join(str(error) for error in source_errors), code="invalid_source",
            repair_paths=tuple(path for error in source_errors for path in error.repair_paths),
        )

    calculated: set[str] = set()
    fields = {name: _prepare_field(name, field, indexed, calculated)
              for name, field in proposal.fields.items()}

    if "vat_mode" in fields:
        if fields["vat_mode"]["value"] == "no_vat" and fields["vat_rate_percent"]["status"] == "fillable":
            raise ExtractionError("Без НДС не должно одновременно иметь заполненную ставку НДС",
                                  repair_paths=("vat_mode", "vat_rate_percent"))
    try:
        return model.model_validate(fields)
    except ValidationError as error:
        raise ExtractionError(
            "Значения не соответствуют типам полей выбранного шаблона",
            repair_paths=tuple(sorted({item["loc"][0] for item in error.errors()
                                       if item["loc"] and item["loc"][0] in proposal.fields})),
        ) from error


def amount_depends_on_unverified_items(proposal: ExtractionProposal, bad_roots: set[str]) -> bool:
    """Вычисляемый итог зависит от позиций, которые не удалось подтвердить.

    Формулы с цитатами не содержат явного графа зависимостей, поэтому итог
    с calculation консервативно считается зависящим от позиций.
    """
    amount = proposal.fields.get("amount_rub")
    return ("items" in bad_roots and "amount_rub" not in bad_roots
            and amount is not None and amount.calculation is not None)


def dependent_amount_issue(stage: str) -> ValidationIssue:
    return ValidationIssue(
        path="amount_rub", stage=stage, code="unverified_dependency",
        message="Вычисляемая сумма требует проверки: данные позиций не подтверждены.",
    )


def prepare_partial_fields(
    proposal: ExtractionProposal,
    messages: list[DocumentMessage],
    document_type: SupportedDocumentType,
) -> tuple[PreparedFields, DocumentValidation]:
    """Сохраняет кандидат и проверяет независимые поля после исчерпания ремонта.

    Строгий prepare_fields по-прежнему отклоняет ошибки. Здесь плохой источник
    блокирует проверку своего поля, остальные поля проверяются обычным кодом.
    Для проблемных полей сохраняются исходные значения, цитаты и планы расчётов.
    """
    indexed = message_index(messages)
    model = FIELDS_BY_TYPE[document_type]
    if set(proposal.fields) != set(model.model_fields):
        raise ExtractionError("Предложение должно содержать ровно все поля выбранного шаблона")

    report = DocumentValidation()
    fields = {}
    calculated = set()
    bad_sources = set()

    def record(error: ExtractionError) -> None:
        for path in error.repair_paths:
            report.issues.append(ValidationIssue(
                path=path, stage="python", code=error.code, message=str(error),
            ))

    for source, path in _proposal_sources(proposal):
        try:
            _check_source(source, indexed, path=path)
        except ExtractionError as error:
            record(error)
            bad_sources.add(field_root(path))

    for name, field in proposal.fields.items():
        checked = _raw_field(field)
        before = calculated.copy()
        if name not in bad_sources:
            try:
                checked = _prepare_field(name, field, indexed, calculated)
                TypeAdapter(model.model_fields[name].annotation).validate_python(checked)
            except ExtractionError as error:
                if not error.repair_paths:
                    error.repair_paths = (name,)
                record(error)
                calculated = before
            except ValidationError:
                record(ExtractionError("Значение не соответствует типу поля", repair_paths=(name,)))
                calculated = before
        fields[name] = checked

    if "vat_mode" in fields and fields["vat_mode"]["value"] == "no_vat" and fields["vat_rate_percent"]["status"] == "fillable":
        record(ExtractionError("Без НДС не должно одновременно иметь заполненную ставку НДС",
                               repair_paths=("vat_mode", "vat_rate_percent")))

    bad_roots = {field_root(issue.path) for issue in report.issues}
    if amount_depends_on_unverified_items(proposal, bad_roots):
        report.issues.append(dependent_amount_issue("python"))
        bad_roots.add("amount_rub")

    for name in bad_roots:
        checked = fields[name]
        checked["validation_status"] = "needs_review"
        try:
            TypeAdapter(model.model_fields[name].annotation).validate_python(checked)
        except ValidationError:
            # Непредставимое в итоговом типе значение остаётся в unverified_fields.
            checked["value"] = None
            checked["derivation"] = None
        report.unverified_fields[name] = proposal.fields[name].model_dump(mode="json")
    if report.issues:
        report.status = report.python_status = "needs_review"
    return model.model_validate(fields), report
