import asyncio
from types import SimpleNamespace

import pytest

from src.cases.document_type import DocumentMessage
from src.schemas.document_schemas import InvoiceFields, TransferFields
from src.extraction import ExtractionProposal
from src.pipeline import PipelineError, process_document
from src.verification import build_verification_questions


def transfer_proposal():
    fields = {
        name: {"status": "missing", "value": None, "evidence": []}
        for name in TransferFields.model_fields
    }
    fields["amount_rub"] = {
        "status": "fillable", "value": 100,
        "evidence": [{"message_id": "m1", "quote": "100 рублей"}],
    }
    return ExtractionProposal.model_validate({"fields": fields})


class FakeExtractor:
    def __init__(self):
        self.calls = []

    async def extract(self, messages, document_type, feedback=None, **kwargs):
        self.calls.append((messages, document_type, feedback))
        return transfer_proposal()


class FakeJev:
    def __init__(self, document_type="transfer_draft", reject=False, incomplete=False):
        self.document_type = document_type
        self.reject = reject
        self.incomplete = incomplete
        self.verification_calls = []

    async def ainvoke(self, state, questions, response_model):
        if hasattr(questions, "document_type"):
            return SimpleNamespace(answers=SimpleNamespace(
                document_type=SimpleNamespace(choice=self.document_type, confidence=0.9),
                none_reason=SimpleNamespace(choice="no_request"),
            ))
        self.verification_calls.append((state, questions))
        answers = {
            key: {"noul": 0.95 if self.reject and key.startswith("amount_rub::") else 0.01}
            for key in questions.root
        }
        if self.incomplete:
            answers.pop(next(iter(answers)))
        return response_model.model_validate({"answers": answers})


def messages():
    return [DocumentMessage(id="m1", role="user", text="Переведи себе 100 рублей")]


def test_full_pipeline_none_never_calls_llm():
    extractor = FakeExtractor()
    client = FakeJev("none")
    result = asyncio.run(process_document("C1", messages(), client, extractor))
    assert result.document_type == "none"
    assert result.fields == {}
    assert result.reason
    assert extractor.calls == []
    assert client.verification_calls == []


def test_full_pipeline_extracts_validates_and_verifies_in_one_batch():
    extractor = FakeExtractor()
    client = FakeJev()
    result = asyncio.run(process_document("C1", messages(), client, extractor))
    assert result.fields.amount_rub.value == 100
    assert result.fields.amount_rub.evidence == ["m1"]
    assert result.fields.destination_bank.status == "missing"
    assert result.confidence == 0.9
    assert len(extractor.calls) == 1
    assert len(client.verification_calls) == 1
    state, questions = client.verification_calls[0]
    assert state.messages == messages()
    assert "destination_bank::status_wrong" in questions.root
    assert "amount_rub::unsupported" in questions.root


def test_rejected_extraction_retries_once_and_then_fails_explicitly():
    extractor = FakeExtractor()
    client = FakeJev(reject=True)
    with pytest.raises(PipelineError, match="C1"):
        asyncio.run(process_document("C1", messages(), client, extractor, max_attempts=2))
    assert len(extractor.calls) == 2
    assert "amount_rub" in " ".join(extractor.calls[1][2])


def test_missing_verifier_answer_cannot_implicitly_pass():
    with pytest.raises(PipelineError):
        asyncio.run(process_document("C1", messages(), FakeJev(incomplete=True), FakeExtractor()))


def test_python_rejects_evidence_before_any_semantic_verification():
    class BadExtractor(FakeExtractor):
        async def extract(self, *args, **kwargs):
            result = transfer_proposal()
            result.fields["amount_rub"].evidence[0].message_id = "missing-id"
            return result

    client = FakeJev()
    with pytest.raises(PipelineError):
        asyncio.run(process_document("C1", messages(), client, BadExtractor(), max_attempts=1))
    assert client.verification_calls == []


def test_duplicate_message_ids_are_rejected_before_calls():
    extractor = FakeExtractor()
    with pytest.raises(PipelineError):
        asyncio.run(process_document("C1", messages() * 2, FakeJev(), extractor))
    assert extractor.calls == []


def test_verification_questions_serialize_as_top_level_questions():
    from src.extraction import prepare_fields
    prepared = prepare_fields(transfer_proposal(), messages(), "transfer_draft")
    questions = build_verification_questions(prepared)
    dumped = questions.model_dump(mode="json")
    assert "root" not in dumped
    assert all(question["type"] == "noul" for question in dumped.values())
    assert "assistant" in dumped["amount_rub::unsupported"]["instructions"].lower()


def test_invoice_formula_is_computed_before_jev_sees_it():
    class InvoiceExtractor:
        async def extract(self, *args, **kwargs):
            fields = {name: {"status": "missing", "value": None, "evidence": []}
                      for name in InvoiceFields.model_fields}
            fields["amount_rub"] = {
                "status": "fillable", "value": None,
                "evidence": [{"message_id": "m1", "quote": "3 часа по 4000 рублей"}],
                "calculation": {
                    "expression": "hours * price",
                    "operands": [
                        {"name": "hours", "source": {"message_id": "m1", "quote": "3"}},
                        {"name": "price", "source": {"message_id": "m1", "quote": "4000"}},
                    ],
                },
            }
            return ExtractionProposal.model_validate({"fields": fields})

    client = FakeJev("invoice_draft")
    dialogue = [DocumentMessage(id="m1", role="user", text="Счёт за 3 часа по 4000 рублей")]
    result = asyncio.run(process_document("C1", dialogue, client, InvoiceExtractor()))
    assert result.fields.amount_rub.value == 12000
    state, questions = client.verification_calls[0]
    assert state.checked_fields.amount_rub.value == 12000
    assert state.checked_fields.amount_rub.derivation
    assert "amount_rub::formula_wrong" in questions.root
    assert state.proposal.fields["amount_rub"].calculation.expression == "hours * price"
    assert "proposal.fields.amount_rub.calculation" in questions.root["amount_rub::formula_wrong"].instructions


def test_retry_can_recover_without_reclassifying():
    class RecoveringJev(FakeJev):
        async def ainvoke(self, state, questions, response_model):
            self.reject = len(self.verification_calls) == 0
            return await super().ainvoke(state, questions, response_model)

    client, extractor = RecoveringJev(), FakeExtractor()
    result = asyncio.run(process_document("C1", messages(), client, extractor))
    assert result.fields.amount_rub.value == 100
    assert len(extractor.calls) == len(client.verification_calls) == 2
    assert extractor.calls[1][2]


def test_wrong_missing_status_triggers_retry_too():
    class MissingErrorJev(FakeJev):
        async def ainvoke(self, state, questions, response_model):
            result = await super().ainvoke(state, questions, response_model)
            if hasattr(questions, "root"):
                result.answers["destination_bank::status_wrong"].noul = 0.9
            return result

    with pytest.raises(PipelineError, match="destination_bank::status_wrong"):
        asyncio.run(process_document("C1", messages(), MissingErrorJev(), FakeExtractor()))


def test_fatal_llm_error_stops_without_using_extraction_attempts():
    from src.llm import LLMError

    class UnavailableExtractor(FakeExtractor):
        async def extract(self, *args, **kwargs):
            self.calls.append(True)
            raise LLMError("Ошибка HTTP 401")

    extractor = UnavailableExtractor()
    with pytest.raises(PipelineError, match="401"):
        asyncio.run(process_document("C1", messages(), FakeJev(), extractor, max_attempts=3))
    assert len(extractor.calls) == 1


def test_invalid_model_output_can_be_repaired_with_feedback():
    from src.llm import LLMResponseError

    class RecoveringExtractor(FakeExtractor):
        async def extract(self, messages, document_type, feedback=None, **kwargs):
            self.calls.append(feedback)
            if len(self.calls) == 1:
                raise LLMResponseError("Некорректный JSON")
            return transfer_proposal()

    extractor = RecoveringExtractor()
    result = asyncio.run(process_document("C1", messages(), FakeJev(), extractor))
    assert result.fields.amount_rub.value == 100
    assert extractor.calls == [[], ["Некорректный JSON"]]


@pytest.mark.parametrize("debug", [False, True])
def test_debug_reports_attempts_and_python_feedback(debug, capsys):
    class MissingEvidenceExtractor(FakeExtractor):
        async def extract(self, *args, **kwargs):
            proposal = transfer_proposal()
            proposal.fields["amount_rub"].evidence = []
            return proposal

    with pytest.raises(PipelineError):
        asyncio.run(process_document(
            "C1", messages(), FakeJev(), MissingEvidenceExtractor(), debug=debug,
        ))
    printed = capsys.readouterr().out
    assert ("Попытка 1/2" in printed) is debug
    assert ("Попытка 2/2" in printed) is debug
    assert ("для значения или проблемы нужны источники" in printed) is debug
    assert ("Классификация" in printed) is debug


def test_debug_reports_jev_feedback(capsys):
    with pytest.raises(PipelineError):
        asyncio.run(process_document("C1", messages(), FakeJev(reject=True), FakeExtractor(), debug=True))
    assert "amount_rub::unsupported" in capsys.readouterr().out


@pytest.mark.parametrize("accept_retry", [False, True])
def test_jev_retry_and_logs_use_short_feedback_without_prompt(monkeypatch, capsys, accept_retry):
    import src.verification as verification

    monkeypatch.setattr(verification, "_RULES", "GENERAL_RULES_FOR_JEV_ONLY")

    class RejectingJev(FakeJev):
        async def ainvoke(self, state, questions, response_model):
            if not hasattr(questions, "document_type"):
                self.reject = not accept_retry or not self.verification_calls
            return await super().ainvoke(state, questions, response_model)

    client = RejectingJev()
    extractor = FakeExtractor()
    if accept_retry:
        result = asyncio.run(process_document("C1", messages(), client, extractor, debug=True))
        assert result.fields.amount_rub.value == 100
    else:
        with pytest.raises(PipelineError) as error:
            asyncio.run(process_document("C1", messages(), client, extractor, debug=True))
        assert "GENERAL_RULES_FOR_JEV_ONLY" not in str(error.value)
        assert "unsupported by its cited user messages" in str(error.value)

    assert len(extractor.calls) == 2
    feedback = extractor.calls[1][2]
    assert any("amount_rub::unsupported" in remark and "0.95" in remark for remark in feedback)
    assert all("GENERAL_RULES_FOR_JEV_ONLY" not in remark for remark in feedback)
    assert "GENERAL_RULES_FOR_JEV_ONLY" not in capsys.readouterr().out
    assert "GENERAL_RULES_FOR_JEV_ONLY" in client.verification_calls[0][1].root["amount_rub::unsupported"].instructions
