"""Разрешённые преобразования названия едины для извлечения и проверки.

Ответы Jev в интеграционных тестах заданы явно, качество реальной модели не оценивается.
"""

import asyncio
import json

import httpx2
import pytest

from scripts.measure_request_sizes import jev_body
from src.cases.document_type import DocumentMessage
from src.extraction import prepare_fields
from src.field_rules import QUOTED_PRODUCT_NAME_RULE
from src.llm import AsyncLLMExtractor, build_extraction_prompt
from src.pipeline import DocumentTypeDecision, process_selected_document
from src.settings import LLMSettings
from src.verification import VerificationResponse, VerificationState, _RULES, build_verification_questions
from tests.test_quoted_product_units import proposal


def inputs(source_name='товар «Подарочный набор "Весна"»', name='Подарочный набор "Весна"', description=None):
    messages = [DocumentMessage(id="m1", role="user", text=f"Счёт за {source_name}: 2 комплекта по 150 рублей.")]
    candidate = proposal(messages, subject=name)
    candidate.fields["items"].value[0].description = description if description is not None else name
    return messages, candidate


@pytest.mark.parametrize("source_name,name", [
    ('товар «Подарочный набор "Весна"»', 'Подарочный набор "Весна"'),
    ('услуга «Аудит "Экспресс"»', 'Аудит "Экспресс"'),
    ('товар «Товар года»', 'Товар года'),
    ('услуга «Услуга года»', 'Услуга года'),
])
def test_real_wire_shares_the_same_quoted_name_rule_for_every_name_check(source_name, name):
    messages, candidate = inputs(source_name, name)
    before = candidate.model_dump(mode="json")
    fields = prepare_fields(candidate, messages, "invoice_draft")
    questions = build_verification_questions(fields, proposal=candidate, messages=messages)
    payload = json.loads(jev_body(
        VerificationState(messages=messages, document_type="invoice_draft", rules=_RULES), questions, VerificationResponse,
    ))
    expected = [key for key in payload["questions"] if key.startswith(("items::", "subject::", "items[0].description::"))]
    assert "items::unsupported" in expected and "items::incomplete" in expected
    assert "items[0].description::unsupported_characteristics" in expected
    for key in expected:
        instructions = payload["questions"][key]["instructions"]
        assert instructions["rules"].count(QUOTED_PRODUCT_NAME_RULE) == 1
        assert instructions["field"]["evidence"] == before["fields"]["items" if key.startswith("items") else "subject"]["evidence"]
    for key in ("items[0].quantity::value_wrong", "items[0].unit::value_wrong", "items[0].unit_price_rub::value_wrong"):
        assert QUOTED_PRODUCT_NAME_RULE not in payload["questions"][key]["instructions"]["rules"]
    assert QUOTED_PRODUCT_NAME_RULE not in payload["questions"]["recipient_name::status_wrong"]["instructions"]["rules"]
    assert candidate.model_dump(mode="json") == before
    assert payload["state"]["messages"] == [message.model_dump(mode="json") for message in messages]
    assert fields.items.value[0].description == fields.subject.value == name
    assert build_extraction_prompt("invoice_draft").count(QUOTED_PRODUCT_NAME_RULE) == 1
    identity = payload["questions"]["items[0].description::value_wrong"]["criteria"]
    characteristics = payload["questions"]["items[0].description::unsupported_characteristics"]["criteria"]
    assert "permitted quoted-name normalization" in identity["false"]
    assert "permitted quoted-name normalization" in characteristics["true"]
    assert "external labels" in characteristics["false"] and "outer quotes" in characteristics["false"]


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("source_name,name,description", [
    ('товар «Подарочный набор "Весна"»', 'Подарочный набор "Весна"', 'Подарочный набор "Весна"'),
    ('товар «Подарочный набор "Весна"»', 'Подарочный набор "Весна"', 'Подарочный набор "Осень"'),
    ('товар «Подарочный набор "Весна"»', 'Подарочный набор "Весна"', 'Подарочный набор Весна'),
    ('товар «Товар года»', 'Товар года', 'Товар года'),
    ('товар «Товар года»', 'Товар года', 'года'),
])
def test_mock_verdict_accepts_allowed_wrapper_removal_and_repairs_only_an_altered_name(protocol, source_name, name, description):
    messages, initial = inputs(source_name, name, description)
    before = initial.model_dump(mode="json")
    _, corrected = inputs(source_name, name)
    requests = []

    def respond(request):
        payload = json.loads(json.loads(request.content)["messages"][-1]["content"])
        requests.append(payload)
        assert payload["repair_paths"] == ["items[0].description"]
        assert payload["repair_fields"] == ["items"]
        assert "unsupported_characteristics" in payload["feedback"][0]
        content = json.dumps({"fields": {"items": corrected.fields["items"].model_dump(mode="json")}})
        if protocol == "anthropic":
            return httpx2.Response(200, json={"content": [{"type": "text", "text": content}], "stop_reason": "end_turn"})
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]})

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            checked = questions.root["items[0].description::unsupported_characteristics"]
            assert QUOTED_PRODUCT_NAME_RULE in checked.instructions["rules"]
            assert QUOTED_PRODUCT_NAME_RULE in questions.root["items::unsupported"].instructions["rules"]
            return response_model.model_validate({"answers": {
                key: {"noul": .95 if key == "items[0].description::unsupported_characteristics"
                      and checked.instructions["field"]["value"] != name else .01}
                for key in questions.root
            }})

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture", LLM_API_PROTOCOL=protocol,
                               LLM_RESPONSE_FORMAT="text")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await process_selected_document(
                    "quoted-name", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
                    Verifier(), extractor, initial_proposal=initial,
                )

    result = asyncio.run(run())
    assert len(requests) == int(description != name)
    assert result.fields == prepare_fields(corrected, messages, "invoice_draft")
    assert initial.model_dump(mode="json") == before
