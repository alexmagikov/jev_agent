"""Предложения LLM, проверка источников и арифметика без выполнения кода модели."""

import ast
import json
import math
import re
from datetime import date
from decimal import Decimal, DecimalException, localcontext
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ValidationError

from src.cases.document_type import DocumentMessage
from src.schemas.document_schemas import (
    InvoiceFields,
    TransferFields,
)
from src.schemas.extraction_schemas import (
    Calculation,
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


@lru_cache(maxsize=2)
def get_proposal_model(document_type: SupportedDocumentType) -> type[BaseModel]:
    """Типизированный ответ LLM: набор полей и типы значений из итогового шаблона.

    Отдельная модель нужна только на границе с LLM. Дальше пайплайн по-прежнему
    работает с ExtractionProposal и словарём fields; формат результата не меняется.
    """

    return PROPOSAL_MODELS[document_type]


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
    """Создаёт индекс реплик для поиска по ID после проверки уникальности."""
    validate_messages(messages)
    indexed = {message.id: message for message in messages}
    return indexed


def _check_source(source: SourceQuote, indexed: dict[str, DocumentMessage], *, path: str) -> None:
    """Путь состоит из имён полей схемы и индексов, без цитат или модельных ID в ошибке."""
    message = indexed.get(source.message_id)

    if message is None or message.role != "user":
        raise ExtractionError(
            f"{path}.message_id: Источник должен ссылаться на существующую реплику user",
            code="invalid_source",
        )
    
    if not source.quote.strip() or source.quote not in message.text:
        raise ExtractionError(
            f"{path}.quote: Цитата источника отсутствует в указанной реплике. "
            "Скопируй дословный фрагмент исходного messages[message_id].text; "
            "не меняй кавычки, пробелы и пунктуацию.",
            code="invalid_source",
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


def _has_whole_unit_source(
    unit: str, sources: list[SourceQuote], indexed: dict[str, DocumentMessage],
) -> bool:
    """Точная форма в указанных репликах, не часть слова/единицы; перед ней допустимо число.

    Проверка только лексическая: актуальность и принадлежность позиции оценивает Jev.
    Точка после короткой единицы может быть пунктуацией предложения, поэтому здесь
    она не объявляется частью сокращения автоматически.
    """
    if not unit.strip():
        return False
    # Слева запрещены буквы/подчёркивание, но разрешена цифра (2м, 2,5кг).
    # Справа запрещены любые word-символы, включая цифры/степени (м != м2/м²).
    pattern = re.compile(rf"(?<![^\W\d]){re.escape(unit)}(?!\w)")
    return any(pattern.search(indexed[source.message_id].text) for source in sources)


_NUMBER = r"[+-]?(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?"
# Число может примыкать к известной единице, но не к произвольному слову/артикулу.
_ATTACHED_UNIT = (
    r"(?:руб(?:лей|ля|ль)?|р|коп(?:еек|ейки|ейку|ейка)?|шт(?:ук(?:а|и|у)?)?|"
    r"кг|гр|мг|г|т|мл|л|км|см|мм|м[²³23]?|ч(?:ас(?:а|ов)?)?|мин|сек|"
    r"уп(?:ак(?:ов(?:ка|ки|ок))?)?|короб(?:ка|ки|ок)|комплект(?:а|ов)?)"
)
_NUMBERS = re.compile(
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
        for match in _NUMBERS.finditer(quote):
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
            for match in _NUMBERS.finditer(indexed[operand.source.message_id].text)
        ):
            raise ExtractionError("Операнд должен быть целым числовым фрагментом, а не частью числа", code="invalid_operand")
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
            if isinstance(node, ast.Constant) and type(node.value) is int and node.value in (0, 1, 100):
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
            raise ExtractionError("В формуле разрешены только операнды, +, -, *, / и скобки", code="invalid_calculation")

        with localcontext() as context:
            context.prec = 40
            value = evaluate(tree.body)
        if used != set(values):
            raise ExtractionError("Все объявленные операнды должны использоваться в формуле", code="invalid_calculation")
        numeric = float(value)
        if not math.isfinite(numeric) or Decimal(str(numeric)) != value:
            raise ExtractionError("Результат нельзя точно представить в JSON; правило округления не задано", code="invalid_calculation")
    except (SyntaxError, DecimalException, OverflowError, RecursionError) as error:
        raise ExtractionError("Некорректная арифметическая формула", code="invalid_calculation") from error
    operands = ", ".join(f"{name}={value}" for name, value in values.items())
    return numeric, f"{target} = {plan.expression}; {operands}; result={value}"


_REQUISITE_FORMATS = {
    "recipient_inn": (r"(?:[0-9]{10}|[0-9]{12})", "Exactly 10 or 12 ASCII digits."),
    "recipient_account": (r"[0-9]{20}", "Exactly 20 ASCII digits."),
    "recipient_bik": (r"[0-9]{9}", "Exactly 9 ASCII digits."),
    "recipient_email": (r"[^\s@]+@[^\s@]+\.[^\s@]+", "Nonempty local@domain.suffix, no whitespace or extra @."),
}


def _invalid_format(name: str, value: object) -> bool:
    """Возвращает True, если некорректный формат"""
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
                       matches_format=not _invalid_format(name, value))
        if isinstance(value, str):
            context["input_length"] = len(value)
    return context


def requisite_source_candidates(name: str, messages: list[DocumentMessage]) -> list[dict] | None:
    """Только строки допустимого формата; принадлежность и актуальность решает Jev.

    Не извлекаем короткое число из более длинного и не используем реплики
    ассистента. None означает поле без поддерживаемого здесь валидатора.
    """
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
            if identity not in seen and not _invalid_format(name, value):
                candidates.append({"message_id": message.id, "quote": value})
                seen.add(identity)
    return candidates


def prepare_fields(
    proposal: ExtractionProposal,
    messages: list[DocumentMessage],
    document_type: SupportedDocumentType,
) -> PreparedFields:
    """Проверяет источники, вычисляет значения и отличает invalid от ошибок модели."""

    indexed = message_index(messages)
    model = InvoiceFields if document_type == "invoice_draft" else TransferFields
    if set(proposal.fields) != set(model.model_fields):
        raise ExtractionError("Предложение должно содержать ровно все поля выбранного шаблона")
    
    fields: dict[str, dict] = {}
    calculated: set[str] = set()

    def apply_calculation(plan: Calculation, target: str, checked_field: dict, *, source_path: str) -> float:
        if len(calculated) >= 32:
            raise ExtractionError("Допустимо не больше 32 вычислений на предложение")
        
        value, formula = _calculate(plan, indexed, target, source_path=source_path)
        checked_field["derivation"] = "; ".join(filter(None, [checked_field["derivation"], formula]))
        checked_field["evidence"] = list(dict.fromkeys([
            *checked_field["evidence"], *(operand.source.message_id for operand in plan.operands),
        ]))
        calculated.add(target)
        return value

    for name, field in proposal.fields.items():
        for index, source in enumerate(field.evidence):
            _check_source(source, indexed, path=f"{name}.evidence[{index}]")
        if field.status != "fillable" and field.value is not None:
            raise ExtractionError(f"{name}: незаполняемое поле должно иметь value=null")
        if field.status in ("fillable", "invalid", "ambiguous") and not field.evidence:
            raise ExtractionError(f"{name}: для значения или проблемы нужны источники", code="missing_evidence")
        
        checked_field = {
            "status": field.status,
            "value": [
                item.model_dump(exclude={"quantity_calculation", "unit_price_calculation"})
                for item in field.value
            ] if isinstance(field.value, list) else field.value,
            "evidence": list(dict.fromkeys(source.message_id for source in field.evidence)),
            "derivation": None,
        }
        if field.calculation is not None:
            if name != "amount_rub":
                raise ExtractionError(f"{name}: недопустимое поле для вычисления")
            if field.status != "fillable":
                raise ExtractionError("Нельзя вычислять незаполняемое поле")
            if field.value is not None:
                raise ExtractionError(f"{name}: при calculation значение должно быть null", code="conflicting_calculation_value")
            
            checked_field["value"] = apply_calculation(
                field.calculation, name, checked_field, source_path=f"{name}.calculation",
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
                        raise ExtractionError(f"{target}: при расчёте значение должно быть null", code="conflicting_calculation_value")
                    checked_field["value"][index][key] = apply_calculation(
                        plan, target, checked_field, source_path=f"items[{index}].{plan_name}",
                    )
        fields[name] = checked_field

    exact_fields = {
        "recipient_inn", "recipient_email", "recipient_account", "recipient_bik",
        "destination_card_masked", "basis_number",
    }

    for name, field in fields.items():
        if field["status"] != "fillable":
            continue
        value = field["value"]
        quotes = [source.quote for source in proposal.fields[name].evidence]

        if value is None:
            raise ExtractionError(f"{name}: у заполняемого поля нет значения или расчёта", code="missing_value")
        if name in exact_fields and (not isinstance(value, str) or not any(value in q for q in quotes)):
            raise ExtractionError(f"{name}: реквизит должен дословно присутствовать в источнике")
        if name in {"recipient_inn", "recipient_account", "recipient_bik"} and not _has_whole_numeric_source(
            value, proposal.fields[name].evidence, indexed,
        ):
            raise ExtractionError(
                f"{name}: скопируй цифровой реквизит целиком из исходного сообщения; "
                "нельзя обрезать цифры в значении или цитате ради нужной длины. "
                "Если исходный реквизит неверного формата, используй invalid, value=null "
                "и дословную цитату проблемного реквизита.",
                code="invalid_source",
            )
        
        numbers: set[Decimal] = set()
        for source in proposal.fields[name].evidence:
            numbers.update(
                _numbers_in([source.quote]) & _numbers_in([indexed[source.message_id].text])
            )

        if name in ("amount_rub", "vat_rate_percent") and name not in calculated:
            if not isinstance(value, (int, float)) or Decimal(str(value)) not in numbers:
                raise ExtractionError(f"{name}: число не подтверждено источником или формулой", code="unsupported_numeric_value")
            
        if name == "items":
            if not isinstance(value, list) or not value:
                raise ExtractionError("items: ожидается непустой список позиций")
            
            for index, item in enumerate(value):
                if item["unit"] is not None and not _has_whole_unit_source(
                    item["unit"], proposal.fields[name].evidence, indexed,
                ):
                    raise ExtractionError(
                        f"items[{index}].unit: единица измерения не найдена целиком в указанных репликах user. "
                        "Скопируй исходную форму без изменения окончания или обозначения; "
                        "если единица не указана, верни unit=null.",
                        code="invalid_source",
                    )
                for key in ("quantity", "unit_price_rub"):
                    number = item[key]
                    if number is not None and f"items[{index}].{key}" not in calculated:
                        if Decimal(str(number)) not in numbers:
                            raise ExtractionError(f"items[{index}].{key}: неподтверждённое число")
                    if number is not None and number < 0:
                        raise ExtractionError("Отрицательное количество или цена позиции")

        if _invalid_format(name, value):
            field.update(status="invalid", value=None, derivation=None)

    if "vat_mode" in fields:
        if fields["vat_mode"]["value"] == "no_vat" and fields["vat_rate_percent"]["status"] == "fillable":
            raise ExtractionError("Без НДС не должно одновременно иметь заполненную ставку НДС")
    try:
        return model.model_validate(fields)
    except ValidationError as error:
        raise ExtractionError("Значения не соответствуют типам полей выбранного шаблона") from error
