import asyncio
import json
from types import SimpleNamespace

import httpx2
import pytest
from pydantic import SecretStr
from typesafe_sdk import TypeSafeAPITimeoutError, TypeSafeRateLimitError, TypeSafeInternalServerError, SystemOneResponse

import src.client as client_module
import src.main as runner
from src.extraction import ExtractionProposal
from src.llm import LLMUsage
from src.pipeline import PipelineRateLimitError
from src.schemas.document_schemas import TransferFields, TransferResult
from src.settings import PipelineSettings
from src.verification import VerificationResponse


@pytest.mark.parametrize("stage", ["classification", "verification"])
@pytest.mark.parametrize("failure", ["timeout", "rate_limit", "server", "validation"])
def test_sdk_failures_have_safe_batch_semantics(monkeypatch, tmp_path, capsys, stage, failure):
    cases = [{"case_id": name, "split": "dev", "messages": [
        {"id": "m1", "role": "user", "text": name},
    ]} for name in ("C1", "C2")]
    fields = {name: {"status": "missing", "value": None, "evidence": [], "derivation": None}
              for name in TransferFields.model_fields}
    path, labels, output = (tmp_path / name for name in ("cases.jsonl", "labels.jsonl", "out.jsonl"))
    path.write_text("\n".join(json.dumps(row) for row in cases))
    labels.write_text("\n".join(TransferResult(case_id=row["case_id"], document_type="transfer_draft",
                                              fields=fields).model_dump_json() for row in cases))
    calls, closed = [], []

    class SDK:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            closed.append(self)

        async def system_one(self, *, state, questions):
            name = state["messages"][0]["text"]
            current = "classification" if "document_type" in questions else "verification"
            calls.append((name, current))
            if name == "C1" and current == stage:
                if failure == "timeout":
                    raise TypeSafeAPITimeoutError(30)
                if failure == "validation":
                    return VerificationResponse.model_validate({"answers": {"bad": {"noul": 2}}})
                error = TypeSafeRateLimitError if failure == "rate_limit" else TypeSafeInternalServerError
                raise error(429 if failure == "rate_limit" else 503,
                            {"message": "PRIVATE_SERVER_BODY"}, httpx2.Headers())
            if current == "classification":
                answers = {
                    "document_type": {"type": "choice", "choice": "transfer_draft", "confidence": .9,
                                      "probabilities": {"transfer_draft": .9, "invoice_draft": .05, "none": .05}},
                    "none_reason": {"type": "choice", "choice": "no_request", "confidence": .9,
                                    "probabilities": {"no_request": .9, "cancelled": .05, "other_document": .05}},
                }
            else:
                answers = {key: {"type": "noul", "noul": .01} for key in questions}
            return SystemOneResponse.model_validate({"model": "test", "usage": {}, "answers": answers})

    class Extractor:
        usage = LLMUsage()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def extract(self, *args, **kwargs):
            return ExtractionProposal.model_validate({"fields": {
                name: {"status": "missing", "value": None, "evidence": []} for name in TransferFields.model_fields}})

    monkeypatch.setattr(client_module, "AsyncTypeSafeClient", lambda **kwargs: SDK())
    monkeypatch.setattr(runner, "JevClientSettings", lambda: SimpleNamespace(api_key=SecretStr("fake")))
    monkeypatch.setattr(runner, "LLMSettings", lambda: object())
    monkeypatch.setattr(runner, "PipelineSettings", lambda: PipelineSettings(_env_file=None))
    monkeypatch.setattr(runner, "AsyncLLMExtractor", lambda *args, **kwargs: Extractor())
    if failure == "rate_limit":
        with pytest.raises(PipelineRateLimitError):
            asyncio.run(runner.main(path, output, labels, mode="full"))
        assert {name for name, _ in calls} == {"C1"}
        assert not output.exists()
    else:
        asyncio.run(runner.main(path, output, labels, mode="full"))
        assert json.loads(output.read_text())["case_id"] == "C2"
    errors = [json.loads(line) for line in output.with_name("out.errors.jsonl").read_text().splitlines()]
    assert [row["case_id"] for row in errors] == ["C1"]
    assert len(closed) == len(calls)
    assert "PRIVATE_SERVER_BODY" not in json.dumps(errors) + capsys.readouterr().out
