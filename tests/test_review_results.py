"""Документ сохраняется после неудачного ремонта, а ошибки остаются видимыми."""

import asyncio

import pytest
from pydantic import TypeAdapter

from src.extraction import prepare_fields
from src.llm import LLMResponseError
from src.pipeline import DocumentTypeDecision, PipelineRejectionError, process_selected_document
from src.schemas.document_schemas import DocumentResult
from tests.test_source_feedback import inputs


class Extractor:
    def __init__(self, proposal, repair_error=False):
        self.proposal = proposal
        self.calls = []
        self.repair_error = repair_error

    async def extract(self, *args, **kwargs):
        self.calls.append(kwargs)
        if self.repair_error and len(self.calls) > 1:
            raise LLMResponseError("PRIVATE_SERVER_BODY", code="invalid_json")
        return self.proposal.model_copy(deep=True)


class Verifier:
    def __init__(self, reject=False):
        self.reject = reject
        self.calls = 0

    async def ainvoke(self, state, questions, response_model):
        self.calls += 1
        return response_model.model_validate({"answers": {
            key: {"noul": .95 if self.reject and key == "subject::unsupported" else .01}
            for key in questions.root
        }})


def run(messages, extractor, verifier=None, **kwargs):
    return asyncio.run(process_selected_document(
        "partial", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
        verifier, extractor, use_verifier=verifier is not None, **kwargs,
    ))


def test_semantic_rejection_preserves_document_and_marks_only_affected_field():
    messages, proposal = inputs()
    extractor, verifier = Extractor(proposal), Verifier(reject=True)
    result = run(messages, extractor, verifier)
    assert len(extractor.calls) == verifier.calls == 2
    assert result.fields.subject.value == proposal.fields["subject"].value
    assert result.fields.amount_rub.value == 300
    assert result.validation.status == "needs_review"
    assert result.validation.python_status == "passed"
    assert result.validation.semantic_status == "needs_review"
    assert [(issue.path, issue.code, issue.stage) for issue in result.validation.issues] == [
        ("subject", "unsupported", "semantic")
    ]
    assert result.fields.subject.validation_status == "needs_review"
    assert result.fields.amount_rub.validation_status is None


def test_semantic_item_rejection_also_marks_the_amount_calculated_from_items():
    class ItemVerifier(Verifier):
        async def ainvoke(self, state, questions, response_model):
            self.calls += 1
            return response_model.model_validate({"answers": {
                key: {"noul": .95 if key == "items[0].quantity::value_wrong" else .01}
                for key in questions.root
            }})

    messages, proposal = inputs()
    result = run(messages, Extractor(proposal), ItemVerifier())
    assert result.fields.items.validation_status == "needs_review"
    assert result.fields.amount_rub.validation_status == "needs_review"
    assert result.fields.amount_rub.value == 300
    assert ("amount_rub", "unverified_dependency", "semantic") in {
        (issue.path, issue.code, issue.stage) for issue in result.validation.issues
    }
    assert "amount_rub" in result.validation.unverified_fields


def test_python_errors_preserve_values_and_collect_other_independent_errors():
    messages, proposal = inputs()
    proposal.fields["subject"].evidence[0].quote = "PRIVATE_BAD_QUOTE"
    proposal.fields["items"].value[0].unit = "PRIVATE_BAD_UNIT"
    verifier = Verifier()
    result = run(messages, Extractor(proposal), verifier)
    assert result.fields.subject.value == proposal.fields["subject"].value
    assert result.fields.items.value[0].unit == "PRIVATE_BAD_UNIT"
    assert result.fields.amount_rub.value == 300
    assert result.validation.python_status == "needs_review"
    assert result.validation.semantic_status == "not_run"
    assert verifier.calls == 0
    paths = {issue.path for issue in result.validation.issues}
    assert "subject.evidence[0].quote" in paths
    assert "items[0].unit" in paths
    assert "amount_rub" in paths  # вычисление зависит от сомнительных данных позиций
    assert "PRIVATE_BAD_QUOTE" not in " ".join(issue.message for issue in result.validation.issues)
    assert result.validation.unverified_fields["subject"]["evidence"][0]["quote"] == "PRIVATE_BAD_QUOTE"


def test_failed_calculation_keeps_other_fields_and_serializes_explicit_null():
    messages, proposal = inputs()
    proposal.fields["amount_rub"].calculation.expression = "count / (price - price)"
    result = run(messages, Extractor(proposal))
    assert result.fields.subject.value == proposal.fields["subject"].value
    assert result.fields.items.value[0].quantity == 2
    assert result.fields.amount_rub.status == "fillable"
    assert result.fields.amount_rub.value is None
    assert result.fields.amount_rub.derivation is None
    assert result.fields.amount_rub.validation_status == "needs_review"
    assert any(issue.path == "amount_rub" and issue.code == "invalid_calculation"
               for issue in result.validation.issues)
    restored = TypeAdapter(DocumentResult).validate_json(result.model_dump_json())
    assert restored == result
    # Строгая проверка для уже существующих потребителей не меняется.
    with pytest.raises(ValueError):
        prepare_fields(proposal, messages, "invoice_draft")


def test_bad_operand_source_does_not_produce_a_calculated_value():
    messages, proposal = inputs()
    proposal.fields["amount_rub"].calculation.operands[0].source.quote = "999"
    result = run(messages, Extractor(proposal))
    assert result.fields.amount_rub.value is None
    assert result.fields.amount_rub.derivation is None
    assert result.fields.items.value[0].unit_price_rub == 150
    assert any(issue.path == "amount_rub.calculation.operands[0].source.quote"
               for issue in result.validation.issues)


@pytest.mark.parametrize("python_error", [False, True])
def test_unparseable_repair_keeps_previous_candidate_and_prior_issues(python_error):
    messages, proposal = inputs()
    if python_error:
        proposal.fields["subject"].evidence[0].quote = "PRIVATE_BAD_QUOTE"
    result = run(messages, Extractor(proposal, repair_error=True), Verifier(reject=not python_error))
    assert result.fields.subject.value == proposal.fields["subject"].value
    assert result.validation.status == "needs_review"
    assert any(issue.stage == ("python" if python_error else "semantic")
               for issue in result.validation.issues)
    assert any(issue.stage == "llm_response" for issue in result.validation.issues)
    assert "PRIVATE_SERVER_BODY" not in result.model_dump_json()


def test_no_parseable_candidate_still_raises():
    messages, _ = inputs()

    class Broken:
        async def extract(self, *args, **kwargs):
            raise LLMResponseError("invalid JSON", code="invalid_json")

    with pytest.raises(PipelineRejectionError):
        run(messages, Broken())


def test_successful_repair_clears_prior_warnings():
    messages, correct = inputs()
    bad = correct.model_copy(deep=True)
    bad.fields["subject"].evidence[0].quote = "PRIVATE_BAD_QUOTE"

    class Recovering(Extractor):
        async def extract(self, *args, **kwargs):
            result = await super().extract(*args, **kwargs)
            return result if len(self.calls) == 1 else correct.model_copy(deep=True)

    result = run(messages, Recovering(bad), Verifier())
    assert result.validation.status == "passed"
    assert result.validation.issues == []
    assert result.validation.unverified_fields == {}
    assert result.fields == prepare_fields(correct, messages, "invoice_draft")


def test_audits_can_explicitly_keep_strict_rejection():
    messages, proposal = inputs()
    with pytest.raises(PipelineRejectionError):
        run(messages, Extractor(proposal), Verifier(reject=True), allow_partial=False)


def test_unverified_field_requires_document_warning_on_deserialization():
    from src.schemas.document_schemas import InvoiceResult

    messages, proposal = inputs()
    fields = prepare_fields(proposal, messages, "invoice_draft").model_dump(mode="json")
    fields["amount_rub"].update(value=None, validation_status="needs_review")
    with pytest.raises(ValueError):
        InvoiceResult(case_id="unsafe", document_type="invoice_draft", fields=fields)


def test_wrong_value_type_is_preserved_in_original_candidate():
    messages, proposal = inputs()
    proposal.fields["recipient_name"].status = "fillable"
    proposal.fields["recipient_name"].value = 150
    proposal.fields["recipient_name"].evidence = proposal.fields["subject"].evidence
    result = run(messages, Extractor(proposal))
    assert result.fields.recipient_name.value is None
    assert result.validation.unverified_fields["recipient_name"]["value"] == 150
    assert result.validation.python_status == "needs_review"


def test_vat_conflict_preserves_both_values_for_review():
    from src.cases.document_type import DocumentMessage
    from src.schemas.extraction_schemas import SourceQuote

    messages, proposal = inputs()
    messages.append(DocumentMessage(id="m3", role="user", text="Без НДС. Ставка 20%"))
    proposal.fields["vat_mode"].status = "fillable"
    proposal.fields["vat_mode"].value = "no_vat"
    proposal.fields["vat_mode"].evidence = [SourceQuote(message_id="m3", quote="Без НДС")]
    proposal.fields["vat_rate_percent"].status = "fillable"
    proposal.fields["vat_rate_percent"].value = 20
    proposal.fields["vat_rate_percent"].evidence = [SourceQuote(message_id="m3", quote="20%")]
    result = run(messages, Extractor(proposal))
    assert result.fields.vat_mode.value == "no_vat"
    assert result.fields.vat_rate_percent.value == 20
    assert result.fields.vat_mode.validation_status == "needs_review"
    assert result.fields.vat_rate_percent.validation_status == "needs_review"


class Sequence:
    """Возвращает заранее заданных кандидатов по порядку вызовов."""

    def __init__(self, *proposals):
        self.proposals = list(proposals)
        self.calls = 0

    async def extract(self, *args, **kwargs):
        self.calls += 1
        return self.proposals[self.calls - 1].model_copy(deep=True)


class FlagsPerCall:
    """На каждом вызове отмечает свой набор проверок как ошибку."""

    def __init__(self, *flags):
        self.flags = flags
        self.calls = 0

    async def ainvoke(self, state, questions, response_model):
        flagged = self.flags[self.calls]
        self.calls += 1
        return response_model.model_validate({"answers": {
            key: {"noul": .95 if key in flagged else .01} for key in questions.root
        }})


def test_python_valid_candidate_wins_over_later_broken_repair():
    messages, first = inputs()
    broken = first.model_copy(deep=True)
    broken.fields["subject"].evidence[0].quote = "PRIVATE_BAD_QUOTE"
    result = run(messages, Sequence(first, broken), FlagsPerCall({"subject::unsupported"}))
    assert result.validation.python_status == "passed"
    assert result.validation.semantic_status == "needs_review"
    assert [(issue.path, issue.stage) for issue in result.validation.issues] == [("subject", "semantic")]
    assert result.validation.unverified_fields["subject"]["evidence"][0]["quote"] == 'Подарок "Май"'


def test_candidate_with_fewer_flagged_fields_wins():
    messages, first = inputs()
    second = first.model_copy(deep=True)
    second.fields["subject"].value = "Подарок"
    verifier = FlagsPerCall({"subject::unsupported"}, {"subject::unsupported", "amount_rub::status_wrong"})
    result = run(messages, Sequence(first, second), verifier)
    assert verifier.calls == 2
    assert result.fields.subject.value == 'Подарок "Май"'
    assert {issue.path for issue in result.validation.issues} == {"subject"}
    assert result.fields.amount_rub.validation_status is None


def test_equally_good_candidates_keep_the_latest_repair():
    messages, first = inputs()
    second = first.model_copy(deep=True)
    second.fields["subject"].value = "Подарок"
    result = run(messages, Sequence(first, second), FlagsPerCall({"subject::unsupported"}, {"subject::unsupported"}))
    assert result.fields.subject.value == "Подарок"
