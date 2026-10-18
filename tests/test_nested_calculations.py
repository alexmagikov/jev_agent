import asyncio
import json
from types import SimpleNamespace

import httpx2
import pytest
from pydantic import ValidationError

from src.cases.document_type import DocumentMessage
from src.schemas.document_schemas import InvoiceFields
from src.extraction import ExtractionError, ExtractionProposal, prepare_fields
from src.llm import AsyncLLMExtractor
from src.main import write_results
from src.pipeline import process_document
from src.settings import LLMSettings
from src.verification import build_verification_questions


def calculation(expression="total / quantity", **operands):
    operands = operands or {"total": "100", "quantity": "5"}
    return {
        "expression": expression,
        "operands": [{"name": name, "source": {"message_id": "m1", "quote": quote}}
                     for name, quote in operands.items()],
    }


def proposal(**fields):
    result = {name: {"status": "missing", "value": None, "evidence": []}
              for name in InvoiceFields.model_fields}
    result.update(fields)
    return {"fields": result}


def filled(value, calculation=None):
    return {
        "status": "fillable", "value": value,
        "evidence": [{"message_id": "m1", "quote": "100 рублей за 5 штук"}],
        "calculation": calculation,
    }


def item(**overrides):
    value = {"description": "Яблоки", "quantity": 5, "unit": "штук", "unit_price_rub": None,
             "unit_price_calculation": calculation()}
    value.update(overrides)
    return value


def prepare(data, text="100 рублей за 5 штук"):
    return prepare_fields(
        ExtractionProposal.model_validate(data),
        [DocumentMessage(id="m1", role="user", text=text)], "invoice_draft",
    )


def test_each_item_owns_its_formula_and_final_items_remain_plain():
    text = "Яблоки: 100 рублей за 5 штук. Груши: 240 рублей за 6 штук"
    data = proposal(items=filled([
        item(),
        item(description="Груши", quantity=6,
             unit_price_calculation=calculation(total="240", quantity="6")),
    ]))
    data["fields"]["items"]["evidence"][0]["quote"] = text
    original = ExtractionProposal.model_validate(data)
    fields = prepare_fields(original, [DocumentMessage(id="m1", role="user", text=text)], "invoice_draft")
    assert [position.unit_price_rub for position in fields.items.value] == [20, 40]
    assert original.fields["items"].value[0].unit_price_rub is None
    assert fields.items.value[0].model_dump() == {
        "description": "Яблоки", "quantity": 5, "unit": "штук", "unit_price_rub": 20,
    }
    assert "items[0].unit_price_rub" in fields.items.derivation
    assert "items[1].unit_price_rub" in fields.items.derivation
    assert "calculation" not in fields.model_dump_json()
    question = build_verification_questions(fields).root["items::formula_wrong"].instructions
    assert "unit_price_calculation" in question
    assert "proposal.calculations" not in question


def test_item_quantity_and_price_can_both_have_nested_formulas():
    text = "2 упаковки по 5 штук; общая цена 100 рублей"
    data = proposal(items=filled([item(
        quantity=None,
        quantity_calculation=calculation("packs * size", packs="2", size="5"),
        unit_price_calculation=calculation("total / (packs * size)", total="100", packs="2", size="5"),
    )]))
    data["fields"]["items"]["evidence"][0]["quote"] = text
    result = prepare(data, text)
    assert result.items.value[0].quantity == 10
    assert result.items.value[0].unit_price_rub == 10
    assert "items[0].quantity" in result.items.derivation


def test_top_level_calculation_merges_all_operand_message_ids():
    data = proposal(amount_rub={
        "status": "fillable", "value": None,
        "evidence": [{"message_id": "m1", "quote": "Подготовь счёт за работу"}],
        "calculation": {
            "expression": "hours * price",
            "operands": [
                {"name": "hours", "source": {"message_id": "m2", "quote": "3"}},
                {"name": "price", "source": {"message_id": "m3", "quote": "4000"}},
            ],
        },
    })
    messages = [
        DocumentMessage(id="m1", role="user", text="Подготовь счёт за работу"),
        DocumentMessage(id="m2", role="user", text="3 часа"),
        DocumentMessage(id="m3", role="user", text="По 4000 рублей"),
    ]
    result = prepare_fields(ExtractionProposal.model_validate(data), messages, "invoice_draft")
    assert result.amount_rub.value == 12000
    assert result.amount_rub.evidence == ["m1", "m2", "m3"]
    question = build_verification_questions(result).root["amount_rub::formula_wrong"].instructions
    assert "proposal.fields.amount_rub.calculation" in question


@pytest.mark.parametrize("name,status,value", [
    ("amount_rub", "missing", None),
    ("subject", "fillable", None),
    ("amount_rub", "fillable", 100),
])
def test_invalid_calculation_location_or_conflicting_value_is_rejected(name, status, value):
    data = proposal(**{name: {**filled(value, calculation()), "status": status}})
    with pytest.raises(ExtractionError):
        prepare(data)


def test_item_cannot_supply_both_unit_price_and_its_calculation():
    with pytest.raises(ExtractionError):
        prepare(proposal(items=filled([item(unit_price_rub=20)])))


def test_nested_unsafe_formula_is_rejected():
    with pytest.raises(ExtractionError):
        prepare(proposal(items=filled([item(unit_price_calculation=calculation("total ** quantity"))])))


def test_nested_operand_still_requires_an_exact_user_quote():
    plan = calculation()
    plan["operands"][0]["source"]["quote"] = "999"
    with pytest.raises(ExtractionError, match="Цитата"):
        prepare(proposal(items=filled([item(unit_price_calculation=plan)])))


def test_nested_calculations_keep_the_total_limit():
    accepted = prepare(proposal(items=filled([item() for _ in range(32)])))
    assert len(accepted.items.value) == 32
    with pytest.raises(ExtractionError, match="32"):
        prepare(proposal(items=filled([item() for _ in range(33)])))


def test_nested_calculation_does_not_round_implicitly():
    text = "100 рублей за 3 штуки"
    data = proposal(items=filled([item(
        quantity=3, unit_price_calculation=calculation(total="100", quantity="3"),
    )]))
    data["fields"]["items"]["evidence"][0]["quote"] = text
    with pytest.raises(ExtractionError, match="округления"):
        prepare(data, text)


def test_old_global_calculations_and_targets_are_rejected():
    data = proposal()
    data["calculations"] = []
    with pytest.raises(ValidationError):
        ExtractionProposal.model_validate(data)
    data = proposal(amount_rub=filled(None, {**calculation(), "target": "amount_rub"}))
    with pytest.raises(ValidationError):
        ExtractionProposal.model_validate(data)


def test_nested_llm_response_reaches_verifier_and_final_jsonl(tmp_path):
    text = "Подготовь счёт. Яблоки: 100 рублей за 5 штук. Груши: 240 рублей за 6 штук"
    data = proposal(
        amount_rub=filled(None, calculation("apples + pears", apples="100", pears="240")),
        items=filled([
            item(), item(description="Груши", quantity=6,
                         unit_price_calculation=calculation(total="240", quantity="6")),
        ]),
    )
    data["fields"]["items"]["evidence"][0]["quote"] = text
    checked = []

    def respond(request):
        prompt = json.loads(json.loads(request.content)["messages"][1]["content"])
        schema = prompt["output_schema"]
        assert set(schema["properties"]) == {"fields"}
        assert set(schema["$defs"]["Calculation"]["properties"]) == {"expression", "operands"}
        assert "unit_price_calculation" in schema["$defs"]["ProposedInvoiceItem"]["properties"]
        return httpx2.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps(data)},
        }]})

    class FakeJev:
        async def ainvoke(self, state, questions, response_model):
            if hasattr(questions, "document_type"):
                return SimpleNamespace(answers=SimpleNamespace(
                    document_type=SimpleNamespace(choice="invoice_draft", confidence=0.9),
                    none_reason=SimpleNamespace(choice="no_request"),
                ))
            checked.append(state)
            return response_model.model_validate({"answers": {
                name: {"noul": 0.01} for name in questions.root
            }})

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="test", LLM_MODEL="small")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await process_document(
                    "nested", [DocumentMessage(id="m1", role="user", text=text)], FakeJev(), extractor,
                )

    result = asyncio.run(run())
    assert len(checked) == 1
    assert checked[0].proposal.fields["items"].value[1].unit_price_calculation is not None
    assert checked[0].checked_fields.items.value[1].unit_price_rub == 40
    output = tmp_path / "predictions.jsonl"
    write_results([result], output)
    saved = json.loads(output.read_text())
    assert saved["fields"]["amount_rub"]["value"] == 340
    assert [position["unit_price_rub"] for position in saved["fields"]["items"]["value"]] == [20, 40]
    assert saved["fields"]["items"]["evidence"] == ["m1"]
    assert "calculation" not in output.read_text()
    assert saved["fields"]["basis_date"]["value"] is None
