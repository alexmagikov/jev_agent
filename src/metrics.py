from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from src.schemas.schemas import DocumentType, Probability
from src.field_rules import equivalent_field_values


type JsonRow = Mapping[str, Any]
type FieldKey = tuple[str, str, str]

SUPPORTED_DOCUMENT_TYPES = ("invoice_draft", "transfer_draft")
ALL_DOCUMENT_TYPES = {*SUPPORTED_DOCUMENT_TYPES, "none"}
FIELD_STATUSES = ("fillable", "missing", "invalid", "ambiguous")
STATUS_OUTCOMES = (*FIELD_STATUSES, "no_prediction", "wrong_document_type", "absent_field")


class FieldMetrics(BaseModel):
    """Метрики определения заполняемых полей и их содержимого."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    value_accuracy: float | None
    alias_aware_value_accuracy: float | None = None
    evidence_id_exact_match: float | None
    overfill_rate: float | None
    expected_values: int
    predicted_values: int
    correct_values: int
    alias_aware_correct_values: int
    value_precision: float | None
    value_recall: float | None
    alias_aware_value_recall: float | None
    expected_fields: int
    correct_statuses: int
    correct_status_values: int
    status_accuracy: float | None
    field_accuracy: float | None
    status_confusion: dict[str, dict[str, int]]

    @property
    def evidence_accuracy(self) -> float | None:
        """Старое Python-имя; в JSON используется evidence_id_exact_match."""
        return self.evidence_id_exact_match

    @property
    def unsupported_fill_rate(self) -> float | None:
        """Старое Python-имя; это доля лишних заполнений, а не оценка доказательств."""
        return self.overfill_rate


class DocumentTypeMismatch(BaseModel):
    """Несовпадение типа; predicted=None означает отсутствие предсказания."""

    case_id: str
    expected: DocumentType
    predicted: DocumentType | None
    confidence: Probability | None = None


class AcceptanceMetrics(BaseModel):
    """Принятие результата независимо от правильности его содержимого."""

    accepted_cases: int
    needs_review_cases: int
    unknown_validation_cases: int
    accepted_correct: int
    accepted_incorrect: int
    needs_review_correct: int
    needs_review_incorrect: int
    automatic_acceptance_rate: float | None
    needs_review_rate: float | None
    accepted_error_rate: float | None
    correct_automatic_acceptance_rate: float | None


class MetricsReport(BaseModel):
    """Итог сравнения предсказаний c правильными результатами."""

    metrics_version: int = 4
    total_cases: int
    predicted_cases: int
    prediction_coverage: float | None
    field_metrics_evaluated: bool
    exact_document_correct: int | None
    exact_document_accuracy: float | None
    acceptance: AcceptanceMetrics | None = None
    missing_predictions: list[str]
    document_type_correct: int
    document_type_accuracy: float
    document_type_mismatches: list[DocumentTypeMismatch]
    fields: FieldMetrics
    by_document_type: dict[str, FieldMetrics]


def calculate_metrics(
    labels: Sequence[JsonRow],
    predictions: Sequence[JsonRow],
    *, evaluate_fields: bool = True,
) -> MetricsReport:
    """Сравнивает предсказания с эталонными строками по ``case_id``.

    Поле считается положительным, когда его ``status`` равен ``fillable``.
    Ключ поля включает ``document_type``, поэтому неверная маршрутизация
    одновременно создаёт пропуски для правильного шаблона и лишние заполнения
    для ошибочно выбранного шаблона.

    ``value_accuracy`` использует строгое равенство JSON-значений, а
    ``evidence_id_exact_match`` — равенство множеств идентификаторов сообщений.
    Обе точности считаются только среди true-positive полей. Если таких полей
    нет, значение метрики равно ``None``.
    ``alias_aware_value_accuracy`` дополнительно учитывает только явно разрешённые
    варианты банка и типа счёта перевода; строгую метрику она не заменяет.

    ``value_recall`` учитывает все ожидаемые заполненные значения, включая
    пропущенные поля и результаты. ``value_precision`` — все заполненные значения
    предсказания. ``overfill_rate`` учитывает только лишние заполнения по разметке.
    Статусы оцениваются по всем ожидаемым полям; отсутствие результата, неверный
    шаблон и отсутствие поля — отдельные столбцы, не status=missing.
    Полностью верный документ совпадает по типу, набору полей, статусам и значениям;
    evidence, derivation, confidence и текст reason в это сравнение не входят.
    Результат validation не меняет правильность содержимого. acceptance отдельно
    учитывает явные passed и needs_review, а отсутствие статуса — как неизвестное.
    Доли принятия и правильного принятия используют все кейсы; доля ошибок среди
    принятых — только passed. При нулевом знаменателе доля полей равна None.
    Для classification передаётся evaluate_fields=False.
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

    label_fields = _all_fields(
        labels_by_id, 
        fields_required=True
    )
    predicted_fields = _all_fields(
        predictions_by_id,
        fields_required=False,
    )
    if not evaluate_fields:
        label_fields, predicted_fields = {}, {}
    predicted_types = {case_id: row["document_type"] for case_id, row in predictions_by_id.items()}
    document_correctness = ({case_id: _exact_document(label, predictions_by_id.get(case_id))
                             for case_id, label in labels_by_id.items()} if evaluate_fields else {})
    exact_documents = sum(document_correctness.values()) if evaluate_fields else None

    by_document_type = {
        document_type: _field_metrics(
            _for_document_type(label_fields, document_type),
            _for_document_type(predicted_fields, document_type),
            predicted_types,
        )
        for document_type in SUPPORTED_DOCUMENT_TYPES
    }

    return MetricsReport(
        total_cases=len(labels_by_id),
        predicted_cases=len(predictions_by_id),
        prediction_coverage=_ratio(len(predictions_by_id), len(labels_by_id)),
        field_metrics_evaluated=evaluate_fields,
        exact_document_correct=exact_documents,
        exact_document_accuracy=_ratio(exact_documents, len(labels_by_id)) if evaluate_fields else None,
        acceptance=_acceptance_metrics(predictions_by_id, document_correctness) if evaluate_fields else None,
        missing_predictions=missing_predictions,
        document_type_correct=document_type_correct,
        document_type_accuracy=_divide(
            document_type_correct,
            len(labels_by_id),
        ),
        document_type_mismatches=document_type_mismatches,
        fields=_field_metrics(label_fields, predicted_fields, predicted_types),
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


def _all_fields(
    rows: Mapping[str, JsonRow],
    fields_required: bool,
) -> dict[FieldKey, JsonRow]:
    result: dict[FieldKey, JsonRow] = {}
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
            if field.get("status") not in FIELD_STATUSES:
                raise ValueError(f"Некорректный статус поля {case_id}.{field_name}")
            result[(case_id, document_type, field_name)] = field

    return result


def field_matches(document_type: str, name: str, expected: JsonRow, actual: JsonRow | None,
                  *, allow_aliases: bool = False) -> bool:
    """Общее сравнение поля для метрик и переходов парного аудита."""
    if actual is None or actual.get("status") != expected["status"] or "value" not in actual:
        return False
    if allow_aliases:
        return equivalent_field_values(document_type, name, actual["value"], expected.get("value"))
    return actual["value"] == expected.get("value")


def _exact_document(expected: JsonRow, actual: JsonRow | None) -> bool:
    if actual is None or actual["document_type"] != expected["document_type"]:
        return False
    target, predicted = expected["fields"], actual.get("fields", {})
    return target.keys() == predicted.keys() and all(
        field_matches(expected["document_type"], name, field, predicted[name])
        for name, field in target.items()
    )


def _acceptance_metrics(
    predictions: Mapping[str, JsonRow], correctness: Mapping[str, bool],
) -> AcceptanceMetrics:
    counts = {"passed": [0, 0], "needs_review": [0, 0]}
    unknown = 0
    for case_id, prediction in predictions.items():
        validation = prediction.get("validation")
        status = validation.get("status") if isinstance(validation, Mapping) else None
        if status not in ("passed", "needs_review"):
            unknown += 1
            continue
        counts[status][0 if correctness[case_id] else 1] += 1
    accepted_correct, accepted_incorrect = counts["passed"]
    review_correct, review_incorrect = counts["needs_review"]
    accepted = accepted_correct + accepted_incorrect
    review = review_correct + review_incorrect
    return AcceptanceMetrics(
        accepted_cases=accepted, needs_review_cases=review, unknown_validation_cases=unknown,
        accepted_correct=accepted_correct, accepted_incorrect=accepted_incorrect,
        needs_review_correct=review_correct, needs_review_incorrect=review_incorrect,
        automatic_acceptance_rate=_ratio(accepted, len(correctness)),
        needs_review_rate=_ratio(review, len(correctness)),
        accepted_error_rate=_ratio(accepted_incorrect, accepted),
        correct_automatic_acceptance_rate=_ratio(accepted_correct, len(correctness)),
    )


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
    predicted_types: Mapping[str, str],
) -> FieldMetrics:
    status_confusion = {status: dict.fromkeys(STATUS_OUTCOMES, 0) for status in FIELD_STATUSES}
    correct_statuses = correct_status_values = 0
    for key, expected in labels.items():
        case_id, document_type, name = key
        actual = predictions.get(key)
        if case_id not in predicted_types:
            outcome = "no_prediction"
        elif predicted_types[case_id] != document_type:
            outcome = "wrong_document_type"
        elif actual is None:
            outcome = "absent_field"
        else:
            outcome = actual["status"]
        status_confusion[expected["status"]][outcome] += 1
        correct_statuses += int(outcome == expected["status"])
        correct_status_values += int(field_matches(document_type, name, expected, actual))

    label_keys = {key for key, field in labels.items() if field["status"] == "fillable"}
    prediction_keys = {key for key, field in predictions.items() if field["status"] == "fillable"}
    true_positive_keys = label_keys & prediction_keys

    true_positives = len(true_positive_keys)
    false_positives = len(prediction_keys - label_keys)
    false_negatives = len(label_keys - prediction_keys)

    correct_values = sum(
        field_matches(key[1], key[2], labels[key], predictions[key])
        for key in true_positive_keys
    )
    correct_evidence = sum(
        _evidence(predictions[key]) == _evidence(labels[key])
        for key in true_positive_keys
    )
    alias_aware_correct_values = sum(
        field_matches(key[1], key[2], labels[key], predictions[key], allow_aliases=True)
        for key in true_positive_keys
    )

    predicted_positive_count = true_positives + false_positives
    actual_positive_count = true_positives + false_negatives

    return FieldMetrics(
        expected_fields=len(labels),
        correct_statuses=correct_statuses,
        correct_status_values=correct_status_values,
        status_accuracy=_ratio(correct_statuses, len(labels)),
        field_accuracy=_ratio(correct_status_values, len(labels)),
        status_confusion=status_confusion,
        expected_values=actual_positive_count,
        predicted_values=predicted_positive_count,
        correct_values=correct_values,
        alias_aware_correct_values=alias_aware_correct_values,
        value_precision=_ratio(correct_values, predicted_positive_count),
        value_recall=_ratio(correct_values, actual_positive_count),
        alias_aware_value_recall=_ratio(alias_aware_correct_values, actual_positive_count),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=_ratio(true_positives, predicted_positive_count),
        recall=_ratio(true_positives, actual_positive_count),
        value_accuracy=(
            correct_values / true_positives if true_positives else None
        ),
        alias_aware_value_accuracy=(
            alias_aware_correct_values / true_positives if true_positives else None
        ),
        evidence_id_exact_match=(
            correct_evidence / true_positives if true_positives else None
        ),
        overfill_rate=_ratio(false_positives, predicted_positive_count),
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


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
