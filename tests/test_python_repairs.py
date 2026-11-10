"""Ремонт после Python не должен менять корректные значения вместе с цитатой."""

import asyncio
import json

import httpx2
import pytest

from src.extraction import ExtractionError, prepare_fields
from src.llm import AsyncLLMExtractor
from src.pipeline import DocumentTypeDecision, process_selected_document
from src.repairs import merge_repair, plan_repair, repair_roots
from src.settings import LLMSettings
from tests.test_source_feedback import inputs


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("use_verifier", [False, True])
def test_python_quote_repair_preserves_unit_with_real_extractor(protocol, use_verifier):
    messages, correct = inputs()
    bad = correct.model_copy(deep=True)
    bad.fields["items"].evidence[0].quote = "PRIVATE_BAD_QUOTE"
    requests, verified = [], []

    def respond(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][-1]["content"])
        requests.append(user)
        if len(requests) == 1:
            reply = bad.model_dump(mode="json")
        else:
            # В новом формате LLM выбирает ID; возвращать единицу или всё items не требуется.
            options = user["quote_fragments"][0]
            fragment = next(f for f in options["fragments"] if 'Подарок "Май"' in f["text"])
            reply = {"selections": [{"path": options["path"], "fragment_id": fragment["fragment_id"]}]}
            assert "fields" not in user["output_schema"]["properties"]
        text = json.dumps(reply)
        return httpx2.Response(200, json=(
            {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}
            if protocol == "anthropic" else
            {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}
        ))

    class Verifier:
        async def ainvoke(self, *, state, questions, response_model):
            assert len(requests) == 2
            verified.append(questions)
            return response_model.model_validate({"answers": {
                key: {"noul": .01} for key in questions.root
            }})

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture",
                               LLM_API_PROTOCOL=protocol, LLM_RESPONSE_FORMAT="text")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await process_selected_document(
                    "T005-regression", messages,
                    DocumentTypeDecision(document_type="invoice_draft", confidence=1),
                    Verifier() if use_verifier else None, extractor, use_verifier=use_verifier,
                )

    result = asyncio.run(run())
    assert result.fields == prepare_fields(correct, messages, "invoice_draft")
    assert result.fields.items.value[0].unit == "комплекта"
    assert len(requests) == 2 and len(verified) == int(use_verifier)
    retry = requests[1]
    assert retry["repair_paths"] == ["items.evidence[0].quote"]
    assert retry["repair_fields"] == ["items"]
    assert set(retry["template"]["fields"]) == {"items"}
    assert retry["previous_proposal"] == bad.model_dump(mode="json")
    assert retry["source_quote_errors"]["sources"] == [{
        "path": "items.evidence[0].quote", "message_id": "m1",
        "invalid_quote": "PRIVATE_BAD_QUOTE", "source_text": messages[0].text,
    }]
    assert "previous_checked_fields" not in retry
    assert "PRIVATE_BAD_QUOTE" not in " ".join(retry["feedback"])


def test_quote_error_has_structured_repair_path():
    messages, bad = inputs()
    bad.fields["items"].evidence[0].quote = "PRIVATE_BAD_QUOTE"
    with pytest.raises(ExtractionError) as raised:
        prepare_fields(bad, messages, "invoice_draft")
    assert raised.value.repair_paths == ("items.evidence[0].quote",)


def test_source_merge_replaces_bad_quote_instead_of_appending_it():
    _, correct = inputs()
    bad = correct.model_copy(deep=True)
    bad.fields["items"].evidence[0].quote = "PRIVATE_BAD_QUOTE"
    reply = correct.model_copy(deep=True)
    reply.fields["items"].value[0].unit = "шт"
    reply.fields["amount_rub"].value = 999
    paths = plan_repair(bad, ["items.evidence[0].quote"])
    assert paths == ["items.evidence[0].quote"]
    assert repair_roots(paths) == ["items"]
    merged = merge_repair(bad, reply, paths)
    assert merged == correct
    assert bad.fields["items"].evidence[0].quote == "PRIVATE_BAD_QUOTE"
    assert merge_repair(bad, merged, paths) == merged


@pytest.mark.parametrize("location", ["amount_rub.calculation",
                                       "items[0].quantity_calculation",
                                       "items[0].unit_price_calculation"])
def test_operand_source_can_be_repaired_without_changing_formula_or_values(location):
    messages, correct = inputs()
    if location.startswith("items["):
        plan_name = location.split(".", 1)[1]
        item = correct.fields["items"].value[0]
        setattr(item, "quantity" if plan_name == "quantity_calculation" else "unit_price_rub", None)
        setattr(item, plan_name, correct.fields["amount_rub"].calculation.model_copy(deep=True))
        plan = getattr(item, plan_name)
    else:
        plan = correct.fields["amount_rub"].calculation
    bad = correct.model_copy(deep=True)
    bad_plan = (bad.fields["amount_rub"].calculation if location.startswith("amount") else
                getattr(bad.fields["items"].value[0], plan_name))
    bad_plan.operands[0].source.quote = "999"
    path = location + ".operands[0].source.quote"
    with pytest.raises(ExtractionError) as raised:
        prepare_fields(bad, messages, "invoice_draft")
    assert raised.value.repair_paths == (path,)
    reply = correct.model_copy(deep=True)
    reply.fields["items"].value[0].unit = "шт"
    merged = merge_repair(bad, reply, plan_repair(bad, [path]))
    assert merged == correct
    prepare_fields(merged, messages, "invoice_draft")


def test_missing_evidence_repair_preserves_item_attributes():
    messages, correct = inputs()
    bad = correct.model_copy(deep=True)
    bad.fields["items"].evidence = []
    with pytest.raises(ExtractionError) as raised:
        prepare_fields(bad, messages, "invoice_draft")
    assert raised.value.repair_paths == ("items.evidence",)
    reply = correct.model_copy(deep=True)
    reply.fields["items"].value[0].unit = "шт"
    assert merge_repair(bad, reply, ["items.evidence"]) == correct


def test_successive_python_errors_repair_the_latest_candidate():
    messages, correct = inputs()
    bad = correct.model_copy(deep=True)
    bad.fields["subject"].evidence[0].quote = "bad subject quote"
    # Две недословные цитаты теперь ремонтируются вместе. Здесь второй ошибкой
    # остаётся отсутствие evidence, обнаруживаемое после проверки цитат.
    bad.fields["items"].evidence = []
    requests = []

    class Extractor:
        async def extract(self, *args, **kwargs):
            requests.append(kwargs)
            reply = correct.model_copy(deep=True)
            if len(requests) == 1:
                return bad.model_copy(deep=True)
            reply.fields["items"].value[0].unit = "шт"
            return reply

    result = asyncio.run(process_selected_document(
        "successive", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
        None, Extractor(), use_verifier=False, max_attempts=3,
    ))
    assert result.fields == prepare_fields(correct, messages, "invoice_draft")
    assert requests[1]["repair_paths"] == ["subject.evidence[0].quote"]
    assert requests[2]["repair_paths"] == ["items.evidence"]
    assert requests[2]["previous_proposal"].fields["subject"] == correct.fields["subject"]


@pytest.mark.parametrize("path", ["items.evidence[-1].quote", "items.evidence[99].quote",
                                   "items.evidence[0].unknown",
                                   "amount_rub.calculation.operands[99].source.quote"])
def test_invalid_source_repair_paths_are_rejected(path):
    _, base = inputs()
    with pytest.raises(ValueError):
        plan_repair(base, [path])


def test_bad_message_id_allows_repair_of_both_source_parts():
    messages, correct = inputs()
    bad = correct.model_copy(deep=True)
    bad.fields["items"].evidence[0].message_id = "unknown"
    bad.fields["items"].evidence[0].quote = "bad quote"
    with pytest.raises(ExtractionError) as raised:
        prepare_fields(bad, messages, "invoice_draft")
    paths = list(raised.value.repair_paths)
    assert set(paths) == {"items.evidence[0].message_id", "items.evidence[0].quote"}
    assert merge_repair(bad, correct, plan_repair(bad, paths)) == correct


def test_source_repair_rejects_changed_list_size():
    _, base = inputs()
    reply = base.model_copy(deep=True)
    reply.fields["items"].evidence.append(reply.fields["items"].evidence[0].model_copy())
    with pytest.raises(ExtractionError, match="Точечный ремонт источника"):
        merge_repair(base, reply, ["items.evidence[0].quote"])


def test_operand_source_repair_rejects_reordered_operands():
    _, base = inputs()
    reply = base.model_copy(deep=True)
    reply.fields["amount_rub"].calculation.operands.reverse()
    with pytest.raises(ExtractionError, match="Точечный ремонт источника"):
        merge_repair(base, reply, ["amount_rub.calculation.operands[0].source.quote"])


def test_parent_repair_path_subsumes_source_leaves():
    _, base = inputs()
    assert plan_repair(base, ["subject", "subject.evidence[0].quote"]) == ["subject"]
    assert plan_repair(base, ["items.evidence", "items.evidence[0].quote"]) == ["items.evidence"]


def test_python_unit_error_repairs_only_unit():
    messages, correct = inputs()
    bad = correct.model_copy(deep=True)
    bad.fields["items"].value[0].unit = "шт"
    calls = []

    class Extractor:
        async def extract(self, *args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return bad.model_copy(deep=True)
            reply = correct.model_copy(deep=True)
            reply.fields["items"].value[0].quantity = 999
            return reply

    result = asyncio.run(process_selected_document(
        "unit", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
        None, Extractor(), use_verifier=False,
    ))
    assert calls[1]["repair_paths"] == ["amount_rub", "items[0].unit"]
    assert result.fields.items.value[0].quantity == 2
    assert result.fields.items.value[0].unit == "комплекта"


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_invalid_json_still_requests_a_complete_new_candidate(protocol):
    messages, correct = inputs()
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(json.loads(body["messages"][-1]["content"]))
        text = "not valid JSON" if len(requests) == 1 else correct.model_dump_json()
        return httpx2.Response(200, json=(
            {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}
            if protocol == "anthropic" else
            {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}
        ))

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture",
                               LLM_API_PROTOCOL=protocol, LLM_RESPONSE_FORMAT="text")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await process_selected_document(
                    "bad-json", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
                    None, extractor, use_verifier=False,
                )

    result = asyncio.run(run())
    assert result.fields == prepare_fields(correct, messages, "invoice_draft")
    assert len(requests) == 2
    assert "repair_paths" not in requests[1] and "previous_proposal" not in requests[1]
    assert requests[1]["feedback"]
    assert len(requests[1]["template"]["fields"]) == len(correct.fields)
