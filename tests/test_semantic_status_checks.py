"""Разделение детерминированного формата, смысла источников и контекста ремонта."""

import asyncio
import json

import httpx2
import pytest

from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, prepare_fields
from src.llm import AsyncLLMExtractor
from src.pipeline import DocumentTypeDecision, process_selected_document
from src.schemas.document_schemas import InvoiceFields
from src.settings import LLMSettings
from src.verification import build_verification_questions


def proposal(value="4070281000000000046", *, status="fillable"):
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in InvoiceFields.model_fields}
    fields["recipient_account"] = {
        "status": status, "value": value if status == "fillable" else None,
        "evidence": [{"message_id": "m1", "quote": value}],
    }
    return ExtractionProposal.model_validate({"fields": fields})


def test_malformed_requisite_has_semantic_source_checks_but_no_format_reassessment():
    messages = [DocumentMessage(id="m1", role="user", text="Счёт ООО Формат, счёт 4070281000000000046")]
    previous_questions = None
    for candidate in (proposal(), proposal(status="invalid")):
        prepared = prepare_fields(candidate, messages, "invoice_draft")
        questions = build_verification_questions(prepared, proposal=candidate, messages=messages).root
        assert {key for key in questions if key.endswith("::status_wrong")} == {
            f"{name}::status_wrong" for name in InvoiceFields.model_fields
        }
        assert "recipient_account::wrong_role" in questions
        assert "recipient_account::superseded" in questions
        assert "recipient_account::missing_supported" not in questions
        assert "recipient_account::unresolved_alternatives" not in questions
        for key in ("recipient_account::wrong_role", "recipient_account::superseded"):
            claim = questions[key].instructions["field"]
            assert claim["status"] == "invalid" and claim["value"] is None
            assert "valid_source_candidates" not in claim
            assert "format_check" not in claim
        semantic_questions = {key: q for key, q in questions.items() if not key.endswith("::status_wrong")}
        if previous_questions is not None:
            assert semantic_questions == previous_questions  # Legacy checks still use the same checked result.
        previous_questions = semantic_questions


@pytest.mark.parametrize("source", [
    "Это мой счёт 40702810000000000001. Счёт получателя сообщу позже.",
    "Исправь счёт получателя на 40702810000000000001.",
])
def test_format_candidates_preserve_role_and_latest_correction_for_model(source):
    messages = [DocumentMessage(id="m1", role="user", text="Счёт ООО Формат, счёт 4070281000000000046"),
                DocumentMessage(id="m2", role="assistant", text="Счёт 40702810000000000002"),
                DocumentMessage(id="m3", role="user", text=source)]
    candidate = proposal()
    prepared = prepare_fields(candidate, messages, "invoice_draft")
    questions = build_verification_questions(prepared, proposal=candidate, messages=messages).root
    q = questions["recipient_account::missing_supported"]
    assert q.instructions["field"]["valid_source_candidates"] == [
        {"message_id": "m3", "quote": "40702810000000000001"},
    ]
    assert "recipient" in q.instructions["question"]
    assert "latest" in q.instructions["question"]
    assert "recipient_account::wrong_role" in questions
    assert "recipient_account::superseded" in questions
    assert "recipient_account::unresolved_alternatives" not in questions


def test_null_ambiguous_field_still_checks_resolution_but_not_rejects_unresolved_alternatives():
    messages = [DocumentMessage(id="m1", role="user", text=(
        "Счёт получателя 40702810000000000001 или 40702810000000000002, ещё решу."
    ))]
    candidate = proposal("40702810000000000001", status="ambiguous")
    prepared = prepare_fields(candidate, messages, "invoice_draft")
    questions = build_verification_questions(prepared, proposal=candidate, messages=messages).root
    assert "recipient_account::missing_supported" in questions
    assert "recipient_account::unresolved_alternatives" not in questions
    assert "unambiguous" in questions["recipient_account::missing_supported"].instructions["question"]


def test_real_repair_request_contains_checked_fields_and_python_changes_without_losing_plan():
    messages = [DocumentMessage(id="m1", role="user", text="Счёт ООО Формат, счёт 4070281000000000046")]
    initial = proposal()
    requests, verifications = [], []

    def respond(request):
        payload = json.loads(request.content)
        user = json.loads(payload["messages"][1]["content"])
        requests.append((payload["messages"][0]["content"], user))
        repaired = proposal(status="invalid")
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"fields": {"recipient_account": repaired.fields["recipient_account"].model_dump(mode="json")}}),
        }}]})

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            verifications.append(questions)
            return response_model.model_validate({"answers": {
                key: {"noul": .95 if len(verifications) == 1 and key == "recipient_account::wrong_role" else .01}
                for key in questions.root
            }})

    async def run():
        config = LLMSettings(_env_file=None, LLM_API_KEY="offline", LLM_MODEL="fixture",
                             LLM_BASE_URL="https://example.test/v1", LLM_MAX_RETRIES=0)
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(config, client=client) as extractor:
                return await process_selected_document(
                    "fixture", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
                    Verifier(), extractor, initial_proposal=initial,
                )

    result = asyncio.run(run())
    assert result.fields.recipient_account.status == "invalid"
    assert len(requests) == 1 and len(verifications) == 2
    system, user = requests[0]
    assert "previous_checked_fields" in system and "python_changes" in system
    assert user["previous_proposal"]["fields"]["recipient_account"]["status"] == "fillable"
    assert user["previous_checked_fields"]["recipient_account"]["status"] == "invalid"
    assert user["python_changes"]["recipient_account"] == {
        "before": {"status": "fillable", "value": "4070281000000000046"},
        "after": {"status": "invalid", "value": None},
    }
    assert set(user["python_changes"]) == {"recipient_account"}
    assert user["repair_paths"] == ["recipient_account"]


@pytest.mark.parametrize("check,new_status", [("superseded", "fillable"), ("wrong_role", "missing")])
def test_semantic_repair_changes_only_the_flagged_requisite(check, new_status):
    messages = [DocumentMessage(id="m1", role="user", text=(
        "Выставь счёт ООО Получатель. Это мой счёт 4070281000000000046."
    ))]
    if check == "superseded":
        messages[0].text = "Счёт ООО Получатель, счёт получателя 4070281000000000046"
        messages.append(DocumentMessage(id="m2", role="user", text="Замени счёт на 40702810000000000001"))
    initial = proposal()
    initial.fields["recipient_name"].status = "fillable"
    initial.fields["recipient_name"].value = "ООО Получатель"
    initial.fields["recipient_name"].evidence = [initial.fields["recipient_account"].evidence[0].model_copy(
        update={"quote": "ООО Получатель"},
    )]
    calls = []

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            key = "recipient_account::" + check
            return response_model.model_validate({"answers": {
                name: {"noul": .95 if name == key and not calls else .01}
                for name in questions.root
            }})

    class Extractor:
        async def extract(self, *args, **kwargs):
            calls.append(kwargs)
            assert kwargs["repair_paths"] == ["recipient_account"]
            assert kwargs["previous_checked_fields"].recipient_account.status == "invalid"
            field = {"status": new_status, "value": None, "evidence": []}
            if new_status == "fillable":
                field.update(value="40702810000000000001", evidence=[
                    {"message_id": "m2", "quote": "40702810000000000001"},
                ])
            return ExtractionProposal.model_validate({"fields": {"recipient_account": field}})

    result = asyncio.run(process_selected_document(
        "fixture", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
        Verifier(), Extractor(), initial_proposal=initial,
    ))
    assert len(calls) == 1
    assert result.fields.recipient_account.status == new_status
    assert result.fields.recipient_name.value == "ООО Получатель"
    if new_status == "fillable":
        assert result.fields.recipient_account.value == "40702810000000000001"
        assert result.fields.recipient_account.evidence == ["m2"]


def test_missing_field_keeps_a_separate_check_for_missed_source_ambiguity():
    messages = [DocumentMessage(id="m1", role="user", text=(
        "Счёт получателя 40702810000000000001 или 40702810000000000002, ещё решу."
    ))]
    candidate = proposal()
    candidate.fields["recipient_account"].status = "missing"
    candidate.fields["recipient_account"].value = None
    candidate.fields["recipient_account"].evidence = []
    prepared = prepare_fields(candidate, messages, "invoice_draft")
    questions = build_verification_questions(prepared, proposal=candidate, messages=messages).root
    assert "recipient_account::unresolved_alternatives" in questions
    assert "recipient_account::missing_supported" in questions


@pytest.mark.parametrize("name,value", [
    ("recipient_inn", "7701234567890"),
    ("recipient_bik", "0445252250"),
    ("recipient_account", "407028100000000000010"),
])
def test_candidate_filter_does_not_extract_a_valid_prefix_from_a_longer_number(name, value):
    from src.extraction import requisite_source_candidates

    assert requisite_source_candidates(name, [DocumentMessage(id="m1", role="user", text=value)]) == []


def test_checked_fields_require_a_repair_base_before_any_http_request():
    initial = proposal()
    messages = [DocumentMessage(id="m1", role="user", text="Счёт 4070281000000000046")]
    prepared = prepare_fields(initial, messages, "invoice_draft")

    async def run():
        def forbidden(request):
            pytest.fail("Invalid repair context must fail before HTTP")
        settings = LLMSettings(_env_file=None, LLM_API_KEY="offline", LLM_MODEL="fixture")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(forbidden)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                with pytest.raises(ValueError, match="предыдущий кандидат"):
                    await extractor.extract(messages, "invoice_draft", previous_checked_fields=prepared)

    asyncio.run(run())
