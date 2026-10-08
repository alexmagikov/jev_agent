import pytest

from src.metrics import calculate_metrics


def field(status: str, value: object = None, evidence: list[str] | None = None) -> dict:
    return {
        "status": status,
        "value": value,
        "evidence": evidence or [],
        "derivation": None,
    }


def test_calculates_end_to_end_document_and_field_metrics() -> None:
    labels = [
        {
            "case_id": "C1",
            "document_type": "invoice_draft",
            "fields": {
                "recipient_name": field("fillable", "ООО Альфа", ["m1"]),
                "amount_rub": field("missing"),
            },
        },
        {
            "case_id": "C2",
            "document_type": "transfer_draft",
            "fields": {
                "amount_rub": field("fillable", 100, ["m2"]),
                "destination_bank": field("missing"),
            },
        },
        {"case_id": "C3", "document_type": "none", "fields": {}},
    ]
    predictions = [
        {
            "case_id": "C1",
            "document_type": "invoice_draft",
            "fields": {
                "recipient_name": field("fillable", "ООО Альфа", ["m1"]),
                "amount_rub": field("fillable", 10, ["m1"]),
            },
        },
        {
            "case_id": "C2",
            "document_type": "invoice_draft",
            "fields": {
                "recipient_name": field("fillable", "ООО Ошибка", ["m2"]),
            },
        },
        {"case_id": "C3", "document_type": "none", "fields": {}},
    ]

    report = calculate_metrics(labels, predictions)

    assert report.total_cases == 3
    assert report.predicted_cases == 3
    assert report.document_type_correct == 2
    assert report.document_type_accuracy == pytest.approx(2 / 3)
    assert report.fields.true_positives == 1
    assert report.fields.false_positives == 2
    assert report.fields.false_negatives == 1
    assert report.fields.precision == pytest.approx(1 / 3)
    assert report.fields.recall == pytest.approx(1 / 2)
    assert report.fields.unsupported_fill_rate == pytest.approx(2 / 3)
    assert report.fields.value_accuracy == 1.0
    assert report.fields.evidence_accuracy == 1.0
    assert report.by_document_type["transfer_draft"].false_negatives == 1


def test_value_and_evidence_accuracy_use_true_positive_fields() -> None:
    labels = [
        {
            "case_id": "C1",
            "document_type": "invoice_draft",
            "fields": {
                "recipient_name": field("fillable", "ООО Альфа", ["m1"]),
                "amount_rub": field("fillable", 100, ["m1", "m3"]),
            },
        }
    ]
    predictions = [
        {
            "case_id": "C1",
            "document_type": "invoice_draft",
            "fields": {
                "recipient_name": field("fillable", "ООО Альфа", ["m9"]),
                "amount_rub": field("fillable", 200, ["m3", "m1"]),
            },
        }
    ]

    report = calculate_metrics(labels, predictions)

    assert report.fields.value_accuracy == 0.5
    assert report.fields.evidence_accuracy == 0.5


def test_missing_prediction_is_wrong_and_produces_false_negatives() -> None:
    labels = [
        {
            "case_id": "C1",
            "document_type": "invoice_draft",
            "fields": {
                "recipient_name": field("fillable", "ООО Альфа", ["m1"]),
            },
        }
    ]

    report = calculate_metrics(labels, [])

    assert report.predicted_cases == 0
    assert report.missing_predictions == ["C1"]
    assert report.document_type_accuracy == 0.0
    assert report.fields.false_negatives == 1
    assert report.fields.value_accuracy is None


def test_rejects_duplicate_case_ids() -> None:
    duplicate_labels = [
        {"case_id": "C1", "document_type": "none", "fields": {}},
        {"case_id": "C1", "document_type": "none", "fields": {}},
    ]

    with pytest.raises(ValueError, match="C1"):
        calculate_metrics(duplicate_labels, [])
