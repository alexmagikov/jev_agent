import asyncio
import json

import httpx2
import pytest

import src.main as runner
import src.pipeline as pipeline
from src.cases.document_type import DocumentMessage, NONE_REASON_TEXT
from src.llm import AsyncLLMExtractor, LLMError
from src.schemas.document_schemas import InvoiceFields, TransferFields
from src.settings import LLMSettings, PipelineSettings


def messages():
    return [DocumentMessage(id="m1", role="user", text="Счёт за 3 часа по 4000 рублей")]


def proposal(document_type="invoice_draft"):
    model = InvoiceFields if document_type == "invoice_draft" else TransferFields
    fields = {name: {"status": "missing", "value": None, "evidence": [], "calculation": None}
              for name in model.model_fields}
    if document_type == "invoice_draft":
        fields["amount_rub"] = {
            "status": "fillable", "value": None,
            "evidence": [{"message_id": "m1", "quote": "3 часа по 4000 рублей"}],
            "calculation": {
                "expression": "hours * price", "operands": [
                    {"name": "hours", "source": {"message_id": "m1", "quote": "3"}},
                    {"name": "price", "source": {"message_id": "m1", "quote": "4000"}},
                ],
            },
        }
    return {"fields": fields}


def invoice_label(case_id="C1"):
    return {"case_id": case_id, "document_type": "invoice_draft", "fields": {
        name: {"status": "missing", "value": None, "evidence": [], "derivation": None}
        for name in InvoiceFields.model_fields
    }}


def classification(document_type="invoice_draft", none_reason=None):
    return {"document_type": document_type, "confidence": 0.8, "none_reason": none_reason}


def forbid_jev(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("llm-only must not configure or call Jev")

    monkeypatch.setattr(pipeline, "classify_document", forbidden)
    monkeypatch.setattr(pipeline, "verify_fields", forbidden)
    monkeypatch.setattr(runner, "JevClientSettings", forbidden)
    monkeypatch.setattr(runner, "AsyncJev", forbidden)


def settings(protocol="anthropic"):
    return LLMSettings(
        _env_file=None, LLM_API_KEY="fake-test-key", LLM_MODEL="small",
        LLM_API_PROTOCOL=protocol, LLM_BASE_URL="https://example.com/v1",
        LLM_RESPONSE_FORMAT="text" if protocol == "anthropic" else "json_schema",
        LLM_MAX_RETRIES=0,
    )


def response(data, protocol):
    content = json.dumps(data)
    if protocol == "anthropic":
        return {"content": [{"type": "text", "text": content}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 100, "output_tokens": 20}}
    return {"choices": [{"finish_reason": "stop", "message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}}


async def run_case(replies, *, protocol="anthropic", max_attempts=2):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        assert len(requests) <= len(replies), "unexpected paid attempt"
        return httpx2.Response(200, json=response(replies[len(requests) - 1], protocol))

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        async with AsyncLLMExtractor(settings(protocol), client=client) as extractor:
            result = await pipeline.process_document(
                "C1", messages(), None, extractor, use_jev=False, max_attempts=max_attempts,
            )
            return result, requests, extractor.usage


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("document_type", ["invoice_draft", "transfer_draft"])
def test_llm_only_classifies_extracts_and_keeps_python_checks(monkeypatch, protocol, document_type):
    forbid_jev(monkeypatch)
    result, requests, usage = asyncio.run(run_case(
        [classification(document_type), proposal(document_type)], protocol=protocol,
    ))
    assert result.document_type == document_type
    assert result.confidence == 0.8
    if document_type == "invoice_draft":
        assert result.fields.amount_rub.value == 12000
        assert result.fields.amount_rub.evidence == ["m1"]
        assert "hours * price" in result.fields.amount_rub.derivation
    assert len(requests) == 2
    assert usage.requests == usage.usage_reports == 2
    assert usage.input_tokens == 200 and usage.output_tokens == 40
    first_user = json.loads(requests[0]["messages"][-1]["content"])
    assert first_user["messages"] == [message.model_dump() for message in messages()]
    assert "output_schema" in first_user
    second_user = json.loads(requests[1]["messages"][-1]["content"])
    assert second_user["document_type"] == document_type
    assert second_user["feedback"] == []
    if protocol == "openai":
        assert requests[0]["response_format"]["json_schema"]["name"] == "document_classification"
        assert requests[0]["response_format"]["json_schema"]["strict"] is True
        assert set(first_user["output_schema"]["required"]) == {"document_type", "confidence", "none_reason"}


@pytest.mark.parametrize("reason", ["cancelled", "other_document", "no_request"])
def test_llm_only_none_has_reason_and_never_extracts(monkeypatch, reason):
    forbid_jev(monkeypatch)
    result, requests, usage = asyncio.run(run_case([classification("none", reason)]))
    assert result.fields == {}
    assert result.reason == NONE_REASON_TEXT[reason]
    assert len(requests) == usage.requests == 1


def test_llm_only_python_error_retries_extraction_without_reclassifying(monkeypatch):
    forbid_jev(monkeypatch)
    invalid = proposal()
    invalid["fields"]["amount_rub"]["evidence"][0]["message_id"] = "not-a-user-message"
    result, requests, _ = asyncio.run(run_case([classification(), invalid, proposal()]))
    assert result.fields.amount_rub.value == 12000
    retry = json.loads(requests[2]["messages"][-1]["content"])
    assert retry["document_type"] == "invoice_draft"
    assert retry["feedback"]
    assert "Jev" not in " ".join(retry["feedback"])


@pytest.mark.parametrize("invalid", [
    {"document_type": "unsupported", "confidence": 0.8, "none_reason": None},
    classification("none"),
    classification("invoice_draft", "cancelled"),
    {"document_type": "none", "confidence": 2, "none_reason": "cancelled"},
])
def test_llm_only_invalid_classification_gets_feedback_not_a_fake_none(monkeypatch, invalid):
    forbid_jev(monkeypatch)
    result, requests, _ = asyncio.run(run_case([invalid, classification("none", "no_request")]))
    assert result.document_type == "none"
    retry = json.loads(requests[1]["messages"][-1]["content"])
    assert retry["feedback"]


def test_llm_only_bad_classification_is_bounded_and_fails_explicitly(monkeypatch):
    forbid_jev(monkeypatch)
    with pytest.raises(pipeline.PipelineError, match="C1.*классификац"):
        asyncio.run(run_case([{}, {}]))


def test_llm_only_invalid_messages_are_rejected_before_inference(monkeypatch):
    forbid_jev(monkeypatch)

    class ForbiddenExtractor:
        async def classify(self, *args, **kwargs):
            pytest.fail("invalid input must not call LLM")

    with pytest.raises(pipeline.PipelineError):
        asyncio.run(pipeline.process_document("C1", messages() * 2, None, ForbiddenExtractor(), use_jev=False))


def test_llm_only_fatal_classification_error_is_not_retried(monkeypatch):
    forbid_jev(monkeypatch)
    calls = []

    class UnavailableExtractor:
        async def classify(self, *args, **kwargs):
            calls.append(True)
            raise LLMError("Ошибка HTTP 401")

    with pytest.raises(pipeline.PipelineError, match="401"):
        asyncio.run(pipeline.process_document("C1", messages(), None, UnavailableExtractor(), use_jev=False))
    assert len(calls) == 1


def test_llm_only_cli_is_explicit_and_does_not_change_default():
    parser = runner.build_parser()
    assert parser.parse_args([]).mode == "classification"
    assert parser.parse_args(["--mode", "llm-only"]).mode == "llm-only"


@pytest.mark.parametrize("default_output", [False, True])
def test_llm_only_cli_needs_no_jev_key_and_keeps_metrics_and_usage(monkeypatch, tmp_path, capsys, default_output):
    forbid_jev(monkeypatch)
    input_path, labels_path = tmp_path / "cases.jsonl", tmp_path / "labels.jsonl"
    output_path = tmp_path / "llm-only.jsonl"
    monkeypatch.setattr(runner, "DEFAULT_LLM_ONLY_OUTPUT_PATH", output_path)
    input_path.write_text(json.dumps({"case_id": "C1", "split": "dev",
                                     "messages": [message.model_dump() for message in messages()]}))
    labels_path.write_text(json.dumps(invoice_label()))
    monkeypatch.setattr(runner, "LLMSettings", lambda: settings())
    monkeypatch.setattr(runner, "PipelineSettings", lambda: PipelineSettings(_env_file=None))
    replies = [classification(), proposal()]

    def respond(request):
        return httpx2.Response(200, json=response(replies.pop(0), "anthropic"))

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            extractor = AsyncLLMExtractor(settings(), client=client)
            monkeypatch.setattr(runner, "AsyncLLMExtractor", lambda *args, **kwargs: extractor)
            await runner.main(input_path, None if default_output else output_path, labels_path, mode="llm-only")

    asyncio.run(run())
    row = json.loads(output_path.read_text())
    assert row["fields"]["amount_rub"]["value"] == 12000
    assert output_path.with_name("llm-only.errors.jsonl").read_text() == ""
    printed = capsys.readouterr().out
    assert "Успешно: 1; с ошибкой: 0" in printed
    assert "1/1 (100.0%)" in printed
    assert "вход=200" in printed and "выход=40" in printed


def test_llm_only_default_output_does_not_overwrite_jev_results():
    assert runner.DEFAULT_LLM_ONLY_OUTPUT_PATH != runner.DEFAULT_FULL_OUTPUT_PATH


def test_llm_only_continues_after_failed_classification_and_counts_errors(monkeypatch, tmp_path, capsys):
    forbid_jev(monkeypatch)
    input_path, labels_path, output_path = (tmp_path / name for name in ("cases.jsonl", "labels.jsonl", "out.jsonl"))
    input_path.write_text("\n".join(json.dumps({
        "case_id": case_id, "split": "dev", "messages": [message.model_dump() for message in messages()],
    }) for case_id in ("C1", "C2")))
    labels_path.write_text("\n".join(json.dumps(invoice_label(case_id)) for case_id in ("C1", "C2")))
    monkeypatch.setattr(runner, "LLMSettings", lambda: settings())
    monkeypatch.setattr(runner, "PipelineSettings", lambda: PipelineSettings(_env_file=None))
    replies = [{}, {}, classification(), proposal()]

    def respond(request):
        return httpx2.Response(200, json=response(replies.pop(0), "anthropic"))

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            extractor = AsyncLLMExtractor(settings(), client=client)
            monkeypatch.setattr(runner, "AsyncLLMExtractor", lambda *args, **kwargs: extractor)
            await runner.main(input_path, output_path, labels_path, mode="llm-only")

    asyncio.run(run())
    assert json.loads(output_path.read_text())["case_id"] == "C2"
    assert json.loads((tmp_path / "out.errors.jsonl").read_text())["case_id"] == "C1"
    printed = capsys.readouterr().out
    assert "Успешно: 1; с ошибкой: 1" in printed
    assert "1/2 (50.0%)" in printed
    assert "запросов=4" in printed
