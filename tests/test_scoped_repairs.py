"""Python ограничивает ремонт атрибутами, не доверяя соблюдению промпта моделью."""

import asyncio
import json

import httpx2
import pytest

from src.extraction import ExtractionError, ExtractionProposal, prepare_fields
from src.pipeline import DocumentTypeDecision, process_selected_document
from src.repairs import merge_repair, plan_repair
from src.schemas.extraction_schemas import Calculation
from src.llm import AsyncLLMExtractor, LLMResponseError
from src.settings import LLMSettings
from src.cases.document_type import DocumentMessage
from tests.test_item_attribute_checks import candidate, dialogue


def test_attribute_merge_preserves_siblings_other_items_and_unrequested_fields():
    base = candidate(quantity=2, unit="шт", unit_price_rub=280)
    reply = base.model_copy(deep=True)
    reply.fields["items"].value[0].quantity = 3
    reply.fields["items"].value[0].unit = "кг"
    reply.fields["items"].value[1].unit_price_rub = 999
    reply.fields["amount_rub"].value = 999
    result = merge_repair(base, reply, ["items[0].quantity"])
    assert result.fields["items"].value[0].quantity == 3
    assert result.fields["items"].value[0].unit == "шт"
    assert result.fields["items"].value[1] == base.fields["items"].value[1]
    assert result.fields["amount_rub"] == base.fields["amount_rub"]
    assert base.fields["items"].value[0].quantity == 2


@pytest.mark.parametrize("change", ["remove", "reorder", "status"])
def test_attribute_repair_rejects_item_identity_or_shape_changes(change):
    base = candidate(quantity=2)
    reply = base.model_copy(deep=True)
    if change == "remove":
        reply.fields["items"].value.pop()
    elif change == "reorder":
        reply.fields["items"].value.reverse()
    else:
        reply.fields["items"].status = "missing"
        reply.fields["items"].value = None
    with pytest.raises(ExtractionError, match="ремонт"):
        merge_repair(base, reply, ["items[0].quantity"])
    assert merge_repair(base, reply, ["items"]).fields["items"] == reply.fields["items"]


def calculation():
    return Calculation.model_validate({"expression": "a / b", "operands": [
        {"name": "a", "source": {"message_id": "m1", "quote": "280"}},
        {"name": "b", "source": {"message_id": "m1", "quote": "2"}},
    ]})


def test_dependency_closure_includes_calculated_price_and_total_but_not_explicit_total():
    base = candidate(quantity=2)
    base.fields["items"].value[0].unit_price_calculation = calculation()
    base = ExtractionProposal.model_validate(base.model_dump())
    assert plan_repair(base, ["items[0].quantity"]) == [
        "items[0].quantity", "items[0].unit_price_rub",
    ]
    base.fields["amount_rub"].calculation = base.fields["items"].value[0].unit_price_calculation
    assert plan_repair(base, ["items[0].quantity"]) == [
        "amount_rub", "items[0].quantity", "items[0].unit_price_rub",
    ]
    assert plan_repair(base, ["items[0].description"]) == ["items[0].description"]
    assert plan_repair(base, ["items", "items[0].quantity"]) == ["amount_rub", "items"]


@pytest.mark.parametrize("path", ["items[-1].quantity", "items[8].quantity", "items[0].bogus", "bogus"])
def test_unknown_repair_paths_are_rejected(path):
    with pytest.raises(ValueError):
        plan_repair(candidate(), [path])


def test_numeric_value_and_its_plan_are_repaired_together_and_evidence_is_not_lost():
    base = candidate(quantity=2)
    reply = base.model_copy(deep=True)
    reply.fields["items"].value[0].quantity = None
    reply.fields["items"].value[0].quantity_calculation = calculation()
    reply = ExtractionProposal.model_validate(reply.model_dump())
    reply.fields["items"].evidence = []
    result = merge_repair(base, reply, ["items[0].quantity"])
    assert result.fields["items"].value[0].quantity is None
    assert result.fields["items"].value[0].quantity_calculation is not None
    assert result.fields["items"].evidence == base.fields["items"].evidence
    assert merge_repair(base, result, ["items[0].quantity"]) == result


def test_pipeline_enforces_attribute_scope_even_for_an_extractor_ignoring_it():
    base = candidate()
    calls, verified = [], []

    class Extractor:
        async def extract(self, *args, **kwargs):
            calls.append(kwargs)
            reply = base.model_copy(deep=True)
            reply.fields["items"].value[0].unit_price_rub = 280
            reply.fields["items"].value[1].unit_price_rub = 999
            return reply

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            verified.append(questions)
            return response_model.model_validate({"answers": {
                key: {"noul": 0.95 if len(verified) == 1 and
                      key == "items[0].unit_price_rub::missing_supported" else 0.01}
                for key in questions.root
            }})

    result = asyncio.run(process_selected_document(
        "scope", dialogue(), DocumentTypeDecision(document_type="invoice_draft", confidence=0.9),
        Verifier(), Extractor(), initial_proposal=base,
    ))
    assert calls[0]["repair_paths"] == ["items[0].unit_price_rub"]
    assert calls[0]["repair_fields"] == ["items"]
    assert result.fields.items.value[0].unit_price_rub == 280
    assert result.fields.items.value[1].unit_price_rub is None
    assert len(verified) == 2


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("reorder", [False, True])
def test_real_extractor_enforces_scope_before_returning_candidate(protocol, reorder):
    base = candidate()
    captured = []

    def respond(request):
        payload = json.loads(request.content)
        captured.append(json.loads(payload["messages"][-1]["content"]))
        reply = base.model_copy(deep=True)
        reply.fields["items"].value[0].unit_price_rub = 280
        reply.fields["items"].value[1].unit_price_rub = 999
        if reorder:
            reply.fields["items"].value.reverse()
        text = json.dumps({"fields": {"items": reply.fields["items"].model_dump(mode="json")}})
        return httpx2.Response(200, json=(
            {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}
            if protocol == "anthropic" else
            {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}
        ))

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fake", LLM_MODEL="small",
                               LLM_API_PROTOCOL=protocol, LLM_RESPONSE_FORMAT="text")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await extractor.extract(dialogue(), "invoice_draft", previous_proposal=base,
                                               repair_fields=["items"], repair_paths=["items[0].unit_price_rub"])

    if reorder:
        with pytest.raises(LLMResponseError, match="ремонт"):
            asyncio.run(run())
    else:
        result = asyncio.run(run())
        assert result.fields["items"].value[0].unit_price_rub == 280
        assert result.fields["items"].value[1] == base.fields["items"].value[1]
        assert result.fields["amount_rub"] == base.fields["amount_rub"]
    assert captured[0]["repair_paths"] == ["items[0].unit_price_rub"]
    assert list(captured[0]["template"]["fields"]) == ["items"]


def test_dependency_plan_is_idempotent_when_total_and_multiple_items_are_calculated():
    base = candidate(quantity=2)
    for item in base.fields["items"].value:
        item.unit_price_calculation = calculation()
    base.fields["amount_rub"].calculation = calculation()
    paths = plan_repair(base, ["items[0].quantity"])
    assert plan_repair(base, paths) == paths
    assert "amount_rub" in paths and "items[1].unit_price_rub" in paths


def test_pipeline_rechecks_quantity_and_dependent_total_after_scoped_formula_repair():
    messages = [DocumentMessage(id="m1", role="user", text=(
        "Счёт: Кабель, 2 упаковки по 5 штук, цена 280 рублей за штуку. Ещё Адаптер."
    ))]
    base = candidate()
    evidence = [{"message_id": "m1", "quote": messages[0].text}]

    def formula(expression):
        return Calculation.model_validate({"expression": expression, "operands": [
            {"name": name, "source": {"message_id": "m1", "quote": quote}}
            for name, quote in (("packs", "2"), ("size", "5"), ("price", "280"))
            if name in expression
        ]})

    raw = base.model_dump()
    raw["fields"]["items"]["evidence"] = evidence
    raw["fields"]["items"]["value"][0].update(
        quantity=None, quantity_calculation=formula("packs + size").model_dump(),
        unit_price_rub=280,
    )
    raw["fields"]["amount_rub"] = {"status": "fillable", "value": None,
                                    "evidence": evidence,
                                    "calculation": formula("(packs + size) * price").model_dump()}
    base = ExtractionProposal.model_validate(raw)
    assert prepare_fields(base, messages, "invoice_draft").amount_rub.value == 1960
    seen = []

    class Extractor:
        async def extract(self, *args, **kwargs):
            assert kwargs["repair_paths"] == ["amount_rub", "items[0].quantity"]
            result = base.model_copy(deep=True)
            result.fields["items"].value[0].quantity_calculation = formula("packs * size")
            result.fields["items"].value[0].unit_price_rub = 999  # Не разрешено.
            result.fields["amount_rub"].calculation = formula("packs * size * price")
            return result

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            seen.append(questions)
            return response_model.model_validate({"answers": {
                key: {"noul": 0.95 if len(seen) == 1 and key == "items[0].quantity::value_wrong" else 0.01}
                for key in questions.root
            }})

    result = asyncio.run(process_selected_document(
        "derived-total", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=0.9),
        Verifier(), Extractor(), initial_proposal=base,
    ))
    assert result.fields.items.value[0].quantity == 10
    assert result.fields.items.value[0].unit_price_rub == 280
    assert result.fields.amount_rub.value == 2800
    assert len(seen) == 2
    assert seen[1].root["amount_rub::formula_wrong"].instructions["field"]["value"] == 2800
