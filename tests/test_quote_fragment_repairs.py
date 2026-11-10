"""Ремонт выбирает фрагмент; дословный текст подставляет Python, без API."""

import asyncio
import json
from pathlib import Path

import httpx2
import pytest

from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionError, ExtractionProposal, prepare_fields
from src.llm import AsyncLLMExtractor, LLMResponseError
from src.pipeline import DocumentTypeDecision, process_selected_document
from src.repairs import source_repair_context, quote_fragment_options, apply_quote_fragment_selections
from src.schemas.extraction_schemas import SourceFragmentSelection
from src.settings import LLMSettings
from tests.test_source_feedback import inputs


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("use_verifier", [False, True])
@pytest.mark.parametrize("field", ["subject", "items"])
def test_quote_repair_selects_id_and_python_copies_original_text(protocol, use_verifier, field):
    messages, correct = inputs()
    bad = correct.model_copy(deep=True)
    bad.fields[field].evidence[0].quote = 'Подарок «Май»'
    requests, verified, selected = [], [], []

    def respond(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][-1]["content"])
        requests.append(user)
        if len(requests) == 1:
            reply = bad.model_dump(mode="json")
        else:
            options = user["quote_fragments"]
            assert len(options) == 1
            assert options[0]["path"] == f"{field}.evidence[0].quote"
            assert options[0]["message_id"] == "m1"
            fragments = options[0]["fragments"]
            assert all(f["text"] in messages[0].text for f in fragments)
            chosen = next(f for f in fragments if 'Подарок "Май"' in f["text"])
            selected.append(chosen["text"])
            reply = {"selections": [{"path": options[0]["path"], "fragment_id": chosen["fragment_id"]}]}
            assert "fields" not in user["output_schema"]["properties"]
            assert "selections" in user["output_schema"]["properties"]
        text = json.dumps(reply)
        return httpx2.Response(200, json=(
            {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}
            if protocol == "anthropic" else
            {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}
        ))

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            verified.append(questions)
            return response_model.model_validate({"answers": {key: {"noul": .01} for key in questions.root}})

    async def run():
        config = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture",
                             LLM_API_PROTOCOL=protocol, LLM_RESPONSE_FORMAT="text")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(config, client=client) as extractor:
                return await process_selected_document(
                    "fragment-repair", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
                    Verifier() if use_verifier else None, extractor, use_verifier=use_verifier,
                )

    result = asyncio.run(run())
    assert result.fields == prepare_fields(correct, messages, "invoice_draft")
    assert result.fields.items.value[0].unit == "комплекта"
    assert len(requests) == 2 and len(verified) == int(use_verifier)
    assert selected and 'Подарок "Май"' in selected[0]
    assert requests[1]["repair_paths"] == [f"{field}.evidence[0].quote"]


@pytest.mark.parametrize("kind", ["unknown_id", "unknown_path", "missing", "duplicate", "none", "quote", "fields", "cross_path"])
def test_selection_rejects_untrusted_or_incomplete_repair_without_mutating_base(kind):
    messages, bad = inputs()
    for field in ("subject", "items"):
        bad.fields[field].evidence[0].quote = "PRIVATE_BAD_QUOTE"
    before = bad.model_dump(mode="json")
    paths = sorted(["subject.evidence[0].quote", "items.evidence[0].quote"])
    options = quote_fragment_options(source_repair_context(bad, paths, messages))

    def respond(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][-1]["content"])
        assert user["quote_fragments"] == options
        schema = body["response_format"]["json_schema"]["schema"]
        properties = schema["$defs"]["SourceFragmentSelection"]["properties"]
        assert set(properties["path"]["enum"]) == set(paths)
        assert set(properties["fragment_id"]["anyOf"][0]["enum"]) == {
            f["fragment_id"] for entry in options for f in entry["fragments"]
        }
        choices = [{"path": entry["path"], "fragment_id": entry["fragments"][0]["fragment_id"]} for entry in options]
        if kind == "unknown_id":
            choices[0]["fragment_id"] = "PRIVATE_UNKNOWN_FRAGMENT"
        elif kind == "unknown_path":
            choices[0]["path"] = "recipient_name.evidence[0].quote"
        elif kind == "missing":
            choices.pop()
        elif kind == "duplicate":
            choices[1] = choices[0].copy()
        elif kind == "none":
            choices[0]["fragment_id"] = None
        elif kind == "quote":
            choices[0]["quote"] = "PRIVATE_INVENTED_QUOTE"
        elif kind == "cross_path":
            choices[0]["fragment_id"] = options[1]["fragments"][0]["fragment_id"]
        reply = {"fields": {}} if kind == "fields" else {"selections": choices}
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(reply)}}]})

    async def run():
        config = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture", LLM_RESPONSE_FORMAT="json_schema")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(config, client=client) as extractor:
                with pytest.raises(LLMResponseError) as raised:
                    await extractor.extract(messages, "invoice_draft", previous_proposal=bad,
                                            repair_fields=["items", "subject"], repair_paths=paths)
        assert "PRIVATE_BAD_QUOTE" not in str(raised.value)
        assert "PRIVATE_UNKNOWN_FRAGMENT" not in str(raised.value)
        assert "PRIVATE_INVENTED_QUOTE" not in str(raised.value)

    asyncio.run(run())
    assert bad.model_dump(mode="json") == before


@pytest.mark.parametrize("path,number", [
    ("amount_rub.calculation.operands[1].source.quote", "150"),
    ("items[0].quantity_calculation.operands[0].source.quote", "2"),
    ("items[0].unit_price_calculation.operands[0].source.quote", "150"),
])
def test_numeric_source_fragments_preserve_formula_and_select_whole_number(path, number):
    messages, correct = inputs()
    if path.startswith("items["):
        plan_name = path.split(".")[1]
        item = correct.fields["items"].value[0]
        attribute = "quantity" if "quantity" in plan_name else "unit_price_rub"
        setattr(item, attribute, None)
        plan = correct.fields["amount_rub"].calculation.model_copy(deep=True)
        plan.operands = [plan.operands[0 if attribute == "quantity" else 1]]
        plan.expression = plan.operands[0].name
        setattr(item, plan_name, plan)
    bad = correct.model_copy(deep=True)
    bad_plan = (bad.fields["amount_rub"].calculation if path.startswith("amount")
                else getattr(bad.fields["items"].value[0], plan_name))
    bad_plan.operands[1 if path.startswith("amount") else 0].source.quote = "999"
    options = quote_fragment_options(source_repair_context(bad, [path], messages))
    assert [f["text"] for f in options[0]["fragments"]] == ["2", "150"]
    chosen = next(f for f in options[0]["fragments"] if f["text"] == number)
    selected = [SourceFragmentSelection(path=path, fragment_id=chosen["fragment_id"])]
    repaired = apply_quote_fragment_selections(bad, options, selected)
    assert repaired == correct
    assert prepare_fields(repaired, messages, "invoice_draft") == prepare_fields(correct, messages, "invoice_draft")


def test_fragment_boundaries_keep_decimal_numbers_and_original_whitespace():
    text = '  Смола: 0,75 кг.\nКабель: 2.5 м; цена 1499,50 рубля.  '
    context = [{"path": "items.evidence[0].quote", "message_id": "m1", "invalid_quote": "bad", "source_text": text}]
    options = quote_fragment_options(context)
    fragments = options[0]["fragments"]
    assert all(f["text"] in text for f in fragments)
    assert any("0,75" in f["text"] for f in fragments)
    assert any("2.5" in f["text"] for f in fragments)
    assert fragments[-1]["text"] == text
    context[0]["path"] = "amount_rub.calculation.operands[0].source.quote"
    assert [f["text"] for f in quote_fragment_options(context)[0]["fragments"]] == ["0,75", "2.5", "1499,50"]


def test_operand_repair_without_source_numbers_does_not_request_an_invented_fragment():
    _, bad = inputs()
    messages = [DocumentMessage(id="m1", role="user", text="Числа пока не сообщили.")]
    bad.fields["amount_rub"].calculation.operands[0].source.quote = "999"

    def respond(request):
        pytest.fail("There are no numeric fragments to select; API must not be called")

    async def run():
        config = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(config, client=client) as extractor:
                with pytest.raises(LLMResponseError, match="нет допустимых фрагментов"):
                    await extractor.extract(messages, "invoice_draft", previous_proposal=bad,
                                            repair_fields=["amount_rub"],
                                            repair_paths=["amount_rub.calculation.operands[0].source.quote"])

    asyncio.run(run())


def test_t005_repairs_m7_quotes_by_id_without_rewriting_item_or_amount():
    data = Path(__file__).resolve().parents[1] / "data"
    case = next(r for line in (data / "cases_tricky35.jsonl").read_text().splitlines() if (r := json.loads(line))["case_id"] == "T005")
    label = next(r for line in (data / "labels_tricky35.jsonl").read_text().splitlines() if (r := json.loads(line))["case_id"] == "T005")
    messages = [DocumentMessage.model_validate(m) for m in case["messages"]]
    texts = {m.id: m.text for m in messages}
    fields = {name: {"status": f["status"], "value": f["value"], "evidence": [
        {"message_id": mid, "quote": texts[mid]} for mid in f["evidence"]
    ]} for name, f in label["fields"].items()}
    fields["items"]["evidence"] = [
        {"message_id": "m7", "quote": 'В текущем заказе используем исходное название Набор «Тихий вечер».'},
        {"message_id": "m5", "quote": "нужны 2 комплекта вместо трёх"},
        {"message_id": "m3", "quote": "Цена 1499,50 относится к одному комплекту целиком"},
    ]
    fields["amount_rub"].update(value=None, calculation={"expression": "count * price", "operands": [
        {"name": "count", "source": {"message_id": "m5", "quote": "2"}},
        {"name": "price", "source": {"message_id": "m1", "quote": "1499,50"}},
    ]})
    bad = ExtractionProposal.model_validate({"fields": fields})
    requests = []

    def respond(request):
        user = json.loads(json.loads(request.content)["messages"][-1]["content"])
        requests.append(user)
        options = user["quote_fragments"][0]
        fragment = next(f for f in options["fragments"] if f["text"].startswith("В текущем заказе"))
        assert fragment["text"] == 'В текущем заказе используем исходное название Набор "Тихий вечер".'
        reply = {"selections": [{"path": options["path"], "fragment_id": fragment["fragment_id"]}]}
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(reply)}}]})

    async def run():
        config = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(config, client=client) as extractor:
                return await process_selected_document(
                    "T005", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
                    None, extractor, initial_proposal=bad, use_verifier=False,
                )

    result = asyncio.run(run())
    assert len(requests) == 1
    assert result.fields.amount_rub.value == 2999
    assert result.fields.items.value[0].description == 'Набор "Тихий вечер"'
    assert result.fields.items.value[0].unit == "комплекта"
    assert result.fields.items.value[0].quantity == 2
    assert result.fields.items.value[0].unit_price_rub == 1499.5
