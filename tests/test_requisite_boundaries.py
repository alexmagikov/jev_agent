"""Цифровые реквизиты копируются целиком, включая границы исходной цитаты."""
import pytest

from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionError, ExtractionProposal, get_proposal_model, prepare_fields
from src.schemas.document_schemas import InvoiceFields


def prepare(name, value, text, quote):
    fields = {key: {"status": "missing", "value": None, "evidence": []}
              for key in InvoiceFields.model_fields}
    fields[name] = {"status": "fillable", "value": value,
                    "evidence": [{"message_id": "m1", "quote": quote}]}
    typed = get_proposal_model("invoice_draft").model_validate({"fields": fields})
    candidate = ExtractionProposal.model_validate(typed.model_dump())
    return prepare_fields(candidate, [DocumentMessage(id="m1", role="user", text=text)], "invoice_draft")


@pytest.mark.parametrize("name,value", [
    ("recipient_inn", "7701234567"),
    ("recipient_inn", "770123456789"),
    ("recipient_bik", "044525225"),
    ("recipient_account", "40702810000000000001"),
])
@pytest.mark.parametrize("before,after", [("", "8"), ("8", ""), ("8", "9")])
@pytest.mark.parametrize("short_quote", [False, True])
def test_rejects_truncated_requisite_even_when_evidence_is_truncated(name, value, before, after, short_quote):
    full = before + value + after
    text = "Реквизит: " + full
    quote = value if short_quote else text
    with pytest.raises(ExtractionError) as raised:
        prepare(name, value, text, quote)
    assert raised.value.code == "invalid_source"
    assert name in str(raised.value)
    assert full not in str(raised.value)


def test_matching_number_elsewhere_does_not_validate_a_truncated_quote():
    value = "7701234567"
    text = f"Правильный ИНН {value}. Старый ИНН {value}8"
    with pytest.raises(ExtractionError):
        prepare("recipient_inn", value, text, f"Старый ИНН {value}")


@pytest.mark.parametrize("name,value", [
    ("recipient_inn", "7701234567"),
    ("recipient_inn", "770123456789"),
    ("recipient_bik", "044525225"),
    ("recipient_account", "40702810000000000001"),
])
@pytest.mark.parametrize("prefix,suffix", [("ИНН", ","), ("Реквизит: ", "."), ("(", ")")])
def test_accepts_whole_requisite_next_to_label_or_punctuation(name, value, prefix, suffix):
    text = prefix + value + suffix
    result = prepare(name, value, text, value)
    assert getattr(result, name).status == "fillable"
    assert getattr(result, name).value == value


@pytest.mark.parametrize("name,value", [
    ("recipient_inn", "77012345678"),
    ("recipient_bik", "0445252259"),
    ("recipient_account", "407028100000000000019"),
])
def test_full_malformed_requisite_keeps_invalid_status(name, value):
    result = prepare(name, value, "Реквизит: " + value, value)
    assert getattr(result, name).status == "invalid"
    assert getattr(result, name).value is None
    assert getattr(result, name).evidence == ["m1"]
