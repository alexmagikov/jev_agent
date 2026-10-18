from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from src.schemas.schemas import DocumentType, Probability
from src.field_rules import equivalent_field_values


type JsonRow = Mapping[str, Any]
type FieldKey = tuple[str, str, str]

SUPPORTED_DOCUMENT_TYPES = ("invoice_draft", "transfer_draft")
ALL_DOCUMENT_TYPES = {*SUPPORTED_DOCUMENT_TYPES, "none"}


class FieldMetrics(BaseModel):
    """Метрики определения заполняемых полей и их содержимого."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    value_accuracy: float | None
    alias_aware_value_accuracy: float | None = None
    evidence_accuracy: float | None
    unsupported_fill_rate: float


class DocumentTypeMismatch(BaseModel):
    """Несовпадение типа; predicted=None означает отсутствие предсказания."""

    case_id: str
    expected: DocumentType
    predicted: DocumentType | None
    confidence: Probability | None = None


class MetricsReport(BaseModel):
    """Итог сравнения предсказаний c правильными результатами."""

    total_cases: int
    predicted_cases: int
    missing_predictions: list[str]
    document_type_correct: int
    document_type_accuracy: float
    document_type_mismatches: list[DocumentTypeMismatch]
    fields: FieldMetrics
    by_document_type: dict[str, FieldMetrics]


def calculate_metrics(
    labels: Sequence[JsonRow],
    predictions: Sequence[JsonRow],
) -> MetricsReport:
    """Сравнивает предсказания с эталонными строками по ``case_id``.

    Поле считается положительным, когда его ``status`` равен ``fillable``.
    Ключ поля включает ``document_type``, поэтому неверная маршрутизация
    одновременно создаёт пропуски для правильного шаблона и лишние заполнения
    для ошибочно выбранного шаблона.

    ``value_accuracy`` использует строгое равенство JSON-значений, а
    ``evidence_accuracy`` — равенство множеств идентификаторов сообщений.
    Обе точности считаются только среди true-positive полей. Если таких полей
    нет, значение метрики равно ``None``.
    ``alias_aware_value_accuracy`` дополнительно учитывает только явно разрешённые
    варианты банка и типа счёта перевода; строгую метрику она не заменяет.
    """

    labels_by_id = _index_rows(labels, source="labels")
    predictions_by_id = _index_rows(predictions, source="predictions")

    unexpected_predictions = set(predictions_by_id) - set(labels_by_id)
    if unexpected_predictions:
        case_ids = ", ".join(sorted(unexpected_predictions))
        raise ValueError(f"Предсказания содержат неизвестные case_id: {case_ids}")

    missing_predictions = [
        case_id for case_id in labels_by_id if case_id not in predictions_by_id
    ]
    document_type_mismatches = [
        DocumentTypeMismatch(
            case_id=case_id,
            expected=label["document_type"],
            predicted=predictions_by_id.get(case_id, {}).get("document_type"),
            confidence=predictions_by_id.get(case_id, {}).get("confidence"),
        )
        for case_id, label in labels_by_id.items()
        if predictions_by_id.get(case_id, {}).get("document_type")
        != label["document_type"]
    ]
    document_type_correct = len(labels_by_id) - len(document_type_mismatches)

    label_fields = _fillable_fields(
        labels_by_id, 
        fields_required=True
    )
    predicted_fields = _fillable_fields(
        predictions_by_id,
        fields_required=False,
    )

    by_document_type = {
        document_type: _field_metrics(
            _for_document_type(label_fields, document_type),
            _for_document_type(predicted_fields, document_type),
        )
        for document_type in SUPPORTED_DOCUMENT_TYPES
    }

    return MetricsReport(
        total_cases=len(labels_by_id),
        predicted_cases=len(predictions_by_id),
        missing_predictions=missing_predictions,
        document_type_correct=document_type_correct,
        document_type_accuracy=_divide(
            document_type_correct,
            len(labels_by_id),
        ),
        document_type_mismatches=document_type_mismatches,
        fields=_field_metrics(label_fields, predicted_fields),
        by_document_type=by_document_type,
    )


def _index_rows(
    rows: Sequence[JsonRow],
    source: str,
) -> dict[str, JsonRow]:
    indexed: dict[str, JsonRow] = {}
    for row_number, row in enumerate(rows, start=1):
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(
                f"Строка {row_number} в {source} не содержит корректный case_id"
            )
        if case_id in indexed:
            raise ValueError(f"Повторяющийся case_id в {source}: {case_id}")

        document_type = row.get("document_type")
        if document_type not in ALL_DOCUMENT_TYPES:
            raise ValueError(
                f"Некорректный document_type для {case_id}: {document_type!r}"
            )
        indexed[case_id] = row

    return indexed


def _fillable_fields(
    rows: Mapping[str, JsonRow],
    fields_required: bool,
) -> dict[FieldKey, JsonRow]:
    fillable: dict[FieldKey, JsonRow] = {}
    for case_id, row in rows.items():
        if fields_required and "fields" not in row:
            raise ValueError(f"В разметке {case_id} отсутствует объект fields")

        fields = row.get("fields", {})
        if not isinstance(fields, Mapping):
            raise ValueError(f"fields для {case_id} должен быть объектом")

        document_type = str(row["document_type"])
        for field_name, field in fields.items():
            if not isinstance(field_name, str) or not isinstance(field, Mapping):
                raise ValueError(f"Некорректное поле в кейсе {case_id}")
            if field.get("status") == "fillable":
                fillable[(case_id, document_type, field_name)] = field

    return fillable


def _for_document_type(
    fields: Mapping[FieldKey, JsonRow],
    document_type: DocumentType,
) -> dict[FieldKey, JsonRow]:
    return {
        key: field
        for key, field in fields.items()
        if key[1] == document_type
    }


def _field_metrics(
    labels: Mapping[FieldKey, JsonRow],
    predictions: Mapping[FieldKey, JsonRow],
) -> FieldMetrics:
    label_keys = set(labels)
    prediction_keys = set(predictions)
    true_positive_keys = label_keys & prediction_keys

    true_positives = len(true_positive_keys)
    false_positives = len(prediction_keys - label_keys)
    false_negatives = len(label_keys - prediction_keys)

    correct_values = sum(
        predictions[key].get("value") == labels[key].get("value")
        for key in true_positive_keys
    )
    correct_evidence = sum(
        _evidence(predictions[key]) == _evidence(labels[key])
        for key in true_positive_keys
    )
    alias_aware_correct_values = sum(
        equivalent_field_values(key[1], key[2], predictions[key].get("value"), labels[key].get("value"))
        for key in true_positive_keys
    )

    predicted_positive_count = true_positives + false_positives
    actual_positive_count = true_positives + false_negatives

    return FieldMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=_divide(true_positives, predicted_positive_count),
        recall=_divide(true_positives, actual_positive_count),
        value_accuracy=(
            correct_values / true_positives if true_positives else None
        ),
        alias_aware_value_accuracy=(
            alias_aware_correct_values / true_positives if true_positives else None
        ),
        evidence_accuracy=(
            correct_evidence / true_positives if true_positives else None
        ),
        unsupported_fill_rate=_divide(
            false_positives,
            predicted_positive_count,
        ),
    )


def _evidence(field: JsonRow) -> set[str]:
    evidence = field.get("evidence", [])
    if not isinstance(evidence, list) or not all(
        isinstance(message_id, str) for message_id in evidence
    ):
        raise ValueError("evidence должен быть списком строк")
    return set(evidence)


def _divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
