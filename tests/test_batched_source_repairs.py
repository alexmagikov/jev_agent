"""Несколько повреждённых источников исправляются за одну попытку ремонта."""

import asyncio
import json
from pathlib import Path

import httpx2
import pytest

from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionError, ExtractionProposal, prepare_fields
from src.llm import AsyncLLMExtractor
from src.pipeline import DocumentTypeDecision, process_selected_document
from src.schemas.document_schemas import TransferFields
from src.settings import LLMSettings
from tests.test_source_feedback import inputs


def test_all_bad_evidence_quotes_are_reported_without_modifying_values():
    messages, bad = inputs()
    for name in ("subject", "items"):
        bad.fields[name].evidence[0].quote = "PRIVATE_BAD_QUOTE"
    original = bad.model_dump(mode="json")
    with pytest.raises(ExtractionError) as raised:
        prepare_fields(bad, messages, "invoice_draft")
    assert raised.value.code == "invalid_source"
    assert raised.value.repair_paths == ("subject.evidence[0].quote", "items.evidence[0].quote")
    for path in raised.value.repair_paths:
        assert path in str(raised.value)
    assert "PRIVATE_BAD_QUOTE" not in str(raised.value)
    assert "PRIVATE_SOURCE_MARKER" not in str(raised.value)
    assert bad.model_dump(mode="json") == original


def test_evidence_and_operand_sources_are_collected_before_any_calculation(monkeypatch):
    messages, bad = inputs()
    # Даже корректный расчёт в первом поле не должен начаться до проверки всех источников.
    bad.fields = {"amount_rub": bad.fields["amount_rub"],
                  **{name: field for name, field in bad.fields.items() if name != "amount_rub"}}
    bad.fields["subject"].evidence[0].quote = "PRIVATE_BAD_QUOTE"
    item = bad.fields["items"].value[0]
    item.unit_price_rub = None
    item.unit_price_calculation = bad.fields["amount_rub"].calculation.model_copy(deep=True)
    item.unit_price_calculation.operands[0].source.quote = "999"
    item.unit_price_calculation.operands[1].source.message_id = "PRIVATE_UNKNOWN_ID"
    def forbidden(*args, **kwargs):
        pytest.fail("Arithmetic must wait until every source is validated")
    monkeypatch.setattr("src.extraction._calculate", forbidden)
    with pytest.raises(ExtractionError) as raised:
        prepare_fields(bad, messages, "invoice_draft")
    assert raised.value.repair_paths == (
        "subject.evidence[0].quote",
        "items[0].unit_price_calculation.operands[0].source.quote",
        "items[0].unit_price_calculation.operands[1].source.message_id",
        "items[0].unit_price_calculation.operands[1].source.quote",
    )
    assert "PRIVATE_UNKNOWN_ID" not in str(raised.value)
    assert "999" not in str(raised.value)


def test_transfer_sources_are_batched_too():
    messages = [DocumentMessage(id="m1", role="user", text="Переведи себе 100 рублей")]
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in TransferFields.model_fields}
    for name, value in (("amount_rub", 100), ("destination_bank", "банк")):
        fields[name] = {"status": "fillable", "value": value,
                        "evidence": [{"message_id": "m1", "quote": "PRIVATE_BAD_QUOTE"}]}
    with pytest.raises(ExtractionError) as raised:
        prepare_fields(ExtractionProposal.model_validate({"fields": fields}), messages, "transfer_draft")
    assert set(raised.value.repair_paths) == {
        "amount_rub.evidence[0].quote", "destination_bank.evidence[0].quote",
    }


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("use_verifier", [False, True])
def test_t019_repairs_both_normalized_quotes_with_one_request(protocol, use_verifier):
    data = Path(__file__).resolve().parents[1] / "data"
    case = next(r for r in map(json.loads, (data / "cases_tricky35.jsonl").read_text().splitlines()) if r["case_id"] == "T019")
    label = next(r for r in map(json.loads, (data / "labels_tricky35.jsonl").read_text().splitlines()) if r["case_id"] == "T019")
    messages = [DocumentMessage.model_validate(m) for m in case["messages"]]
    texts = {m.id: m.text for m in messages}
    fields = {name: {"status": f["status"], "value": f["value"], "evidence": [
        {"message_id": mid, "quote": texts[mid]} for mid in f["evidence"]
    ]} for name, f in label["fields"].items()}
    correct = ExtractionProposal.model_validate({"fields": fields})
    bad = correct.model_copy(deep=True)
    for name in ("subject", "items"):
        bad.fields[name].evidence[0].quote = "проверка макета"
    original = bad.model_dump(mode="json")
    requests, verifications = [], []
    expected_paths = ["items.evidence[0].quote", "subject.evidence[0].quote"]

    def respond(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][-1]["content"])
        requests.append(user)
        assert user["repair_paths"] == expected_paths
        assert user["repair_fields"] == ["items", "subject"]
        assert set(user["template"]["fields"]) == {"items", "subject"}
        assert user["previous_proposal"] == original
        options = user["quote_fragments"]
        assert [entry["path"] for entry in options] == expected_paths
        choices = [{"path": entry["path"], "fragment_id": next(
            fragment["fragment_id"] for fragment in entry["fragments"] if fragment["text"] == texts["m1"]
        )} for entry in options]
        content = json.dumps({"selections": choices})
        return httpx2.Response(200, json=(
            {"content": [{"type": "text", "text": content}], "stop_reason": "end_turn"}
            if protocol == "anthropic" else
            {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        ))

    class Verifier:
        async def ainvoke(self, *, state, questions, response_model):
            assert len(requests) == 1
            verifications.append(questions)
            return response_model.model_validate({"answers": {key: {"noul": .01} for key in questions.root}})

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture",
                               LLM_API_PROTOCOL=protocol, LLM_RESPONSE_FORMAT="text")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await process_selected_document(
                    "T019", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
                    Verifier() if use_verifier else None, extractor,
                    initial_proposal=bad, max_attempts=2, use_verifier=use_verifier,
                )
    result = asyncio.run(run())
    assert len(requests) == 1 and len(verifications) == int(use_verifier)
    assert result.fields == prepare_fields(correct, messages, "invoice_draft")
    assert bad.model_dump(mode="json") == original
