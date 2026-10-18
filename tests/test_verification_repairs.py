import asyncio
import json
from types import SimpleNamespace

import httpx2

from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, prepare_fields
from src.llm import AsyncLLMExtractor, EXTRACTION_PROMPT
from src.pipeline import process_document
from src.schemas.document_schemas import InvoiceFields
from src.settings import LLMSettings
from src.verification import build_verification_questions


def invoice_proposal(*, name="ИП Смирнову", amount=1500):
    fields = {key: {"status": "missing", "value": None, "evidence": []}
              for key in InvoiceFields.model_fields}
    fields["recipient_name"] = {
        "status": "fillable", "value": name,
        "evidence": [{"message_id": "m1", "quote": "ИП Смирнову"}],
    }
    fields["amount_rub"] = {
        "status": "fillable", "value": amount,
        "evidence": [{"message_id": "m1", "quote": "1500 рублей"}],
    }
    return ExtractionProposal.model_validate({"fields": fields})


def messages():
    return [DocumentMessage(id="m1", role="user", text="Счёт ИП Смирнову на 1500 рублей")]


class RejectNameOnce:
    def __init__(self):
        self.verification_calls = []

    async def ainvoke(self, state, questions, response_model):
        if hasattr(questions, "document_type"):
            return SimpleNamespace(answers=SimpleNamespace(
                document_type=SimpleNamespace(choice="invoice_draft", confidence=0.9),
                none_reason=SimpleNamespace(choice="no_request"),
            ))
        self.verification_calls.append(state)
        return response_model.model_validate({"answers": {
            key: {"noul": 0.95 if len(self.verification_calls) == 1 and key == "recipient_name::noncanonical" else 0.01}
            for key in questions.root
        }})


def test_jev_repairs_only_flagged_fields_and_rechecks_merged_result():
    calls = []

    class Extractor:
        async def extract(self, messages, document_type, feedback=None, **kwargs):
            calls.append((feedback, kwargs))
            if len(calls) == 1:
                return invoice_proposal()
            # Ремонт имени не должен испортить уже проверенную сумму.
            return invoice_proposal(name="ИП Смирнов", amount=9999)

    client = RejectNameOnce()
    result = asyncio.run(process_document("C1", messages(), client, Extractor()))
    assert result.fields.recipient_name.value == "ИП Смирнов"
    assert result.fields.amount_rub.value == 1500
    assert len(client.verification_calls) == 2
    assert client.verification_calls[1].checked_fields.amount_rub.value == 1500
    feedback, repair = calls[1]
    assert repair["repair_fields"] == ["recipient_name"]
    assert repair["previous_proposal"].fields["recipient_name"].value == "ИП Смирнову"
    assert any("recipient_name::noncanonical" in remark for remark in feedback)


def test_jev_has_atomic_checks_for_each_known_item_attribute():
    data = invoice_proposal().model_dump()
    data["fields"]["items"] = {
        "status": "fillable", "value": [{
            "description": "папки", "quantity": 3, "unit": None, "unit_price_rub": 500,
        }], "evidence": [{"message_id": "m1", "quote": "3 папки по 500"}],
    }
    proposal = ExtractionProposal.model_validate(data)
    prepared = prepare_fields(proposal, [DocumentMessage(
        id="m1", role="user", text="Счёт ИП Смирнову: 3 папки по 500, всего 1500 рублей",
    )], "invoice_draft")
    questions = build_verification_questions(prepared).root
    for attribute in ("description", "quantity", "unit_price_rub"):
        key = f"items[0].{attribute}::value_wrong"
        assert key in questions
        assert f"checked_fields.items.value[0].{attribute}" in questions[key].instructions
        assert "single line item" in questions[key].instructions
    assert "items[0].unit::value_wrong" not in questions
    assert "items::incomplete" in questions


def test_common_prompt_and_jev_agree_on_canonical_names_and_basis_numbers():
    assert "именительный падеж" in EXTRACTION_PROMPT
    assert "без обозначения №" in EXTRACTION_PROMPT
    prepared = prepare_fields(invoice_proposal(), messages(), "invoice_draft")
    questions = build_verification_questions(prepared).root
    assert "recipient_name::noncanonical" in questions
    assert "nominative" in questions["recipient_name::noncanonical"].instructions
    assert "subject" in EXTRACTION_PROMPT and "не склеивай" in EXTRACTION_PROMPT


def test_atomic_jev_feedback_repairs_a_numeric_role_error_that_python_cannot_detect():
    dialogue = [DocumentMessage(id="m1", role="user", text="Счёт ИП Смирнову: 3 папки по 500, всего 1500 рублей")]

    def candidate(quantity):
        data = invoice_proposal(name="ИП Смирнов").model_dump()
        data["fields"]["items"] = {
            "status": "fillable", "value": [{"description": "папка", "quantity": quantity,
                "unit": None, "unit_price_rub": 500}],
            "evidence": [{"message_id": "m1", "quote": "3 папки по 500"}],
        }
        return ExtractionProposal.model_validate(data)

    # Оба числа встречаются в источнике: одной проверки принадлежности чисел недостаточно.
    assert prepare_fields(candidate(500), dialogue, "invoice_draft").items.value[0].quantity == 500
    repairs = []

    class Extractor:
        async def extract(self, *args, **kwargs):
            repairs.append(kwargs)
            return candidate(500 if len(repairs) == 1 else 3)

    class ItemVerifier(RejectNameOnce):
        async def ainvoke(self, state, questions, response_model):
            if hasattr(questions, "document_type"):
                return await super().ainvoke(state, questions, response_model)
            self.verification_calls.append(state)
            return response_model.model_validate({"answers": {
                key: {"noul": 0.99 if key == "items[0].quantity::value_wrong"
                      and state.checked_fields.items.value[0].quantity == 500 else 0.01}
                for key in questions.root
            }})

    result = asyncio.run(process_document("C1", dialogue, ItemVerifier(), Extractor()))
    assert result.fields.items.value[0].quantity == 3
    assert repairs[1]["repair_fields"] == ["items"]
    assert "items[0].quantity::value_wrong" in " ".join(repairs[1]["feedback"])


def test_llm_repair_payload_contains_candidate_and_allowed_fields_without_labels():
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return httpx2.Response(200, json={
            "content": [{"type": "text", "text": invoice_proposal(name="ИП Смирнов").model_dump_json()}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 5},
        })

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fake", LLM_MODEL="small",
                               LLM_API_PROTOCOL="anthropic", LLM_BASE_URL="https://example.com/v1")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                await extractor.extract(messages(), "invoice_draft", feedback=["recipient_name::noncanonical"],
                                        previous_proposal=invoice_proposal(), repair_fields=["recipient_name"])

    asyncio.run(run())
    user = json.loads(captured[0]["messages"][0]["content"])
    assert user["repair_fields"] == ["recipient_name"]
    assert user["previous_proposal"]["fields"]["recipient_name"]["value"] == "ИП Смирнову"
    assert not {"labels", "expected", "gold"} & user.keys()


def test_offline_size_report_includes_targeted_repair_and_atomic_checks(capsys):
    from scripts.measure_request_sizes import main

    asyncio.run(main())
    stats = json.loads(capsys.readouterr().out)
    assert stats["tokens"] is None
    assert stats["c001"]["verification_question_kinds"]["value_wrong"] > 0
    sizes = stats["c001"]["request_and_example_sizes"]
    assert sizes["07-llm-retry-jev.request.json"]["utf8_bytes"] > sizes["02-llm-extraction.request.json"]["utf8_bytes"]


def test_invoice_amount_has_a_separate_price_total_ambiguity_check():
    from src.field_rules import INVOICE_AMOUNT_RULE

    prepared = prepare_fields(invoice_proposal(), messages(), "invoice_draft")
    questions = build_verification_questions(prepared).root
    key = "amount_rub::ambiguous"
    assert key in questions
    assert "unit price" in questions[key].instructions
    assert INVOICE_AMOUNT_RULE in questions[key].instructions
    assert INVOICE_AMOUNT_RULE in EXTRACTION_PROMPT


def test_subject_and_item_unit_rules_are_shared_by_extractor_and_verifier():
    from src.field_rules import INVOICE_SUBJECT_RULE, ITEM_UNIT_RULE

    data = invoice_proposal().model_dump()
    data["fields"]["items"] = {"status": "fillable", "value": [{
        "description": "ремонт", "quantity": 2, "unit": "ремонта", "unit_price_rub": None,
    }], "evidence": [{"message_id": "m1", "quote": "2 ремонта"}]}
    prepared = prepare_fields(ExtractionProposal.model_validate(data), [DocumentMessage(
        id="m1", role="user", text="Счёт ИП Смирнову: 2 ремонта, всего 1500 рублей",
    )], "invoice_draft")
    questions = build_verification_questions(prepared).root
    assert INVOICE_SUBJECT_RULE in questions["subject::status_wrong"].instructions
    assert ITEM_UNIT_RULE in questions["items[0].unit::value_wrong"].instructions
    assert INVOICE_SUBJECT_RULE in EXTRACTION_PROMPT and ITEM_UNIT_RULE in EXTRACTION_PROMPT


def test_amount_ambiguity_feedback_repairs_status_and_preserves_other_fields():
    dialogue = [DocumentMessage(id="m1", role="user", text="Счёт ИП Смирнову: плитка 1500 рублей")]

    class Extractor:
        async def extract(self, *args, **kwargs):
            data = invoice_proposal(name="ИП Смирнов").model_dump()
            if kwargs.get("repair_fields"):
                assert kwargs["repair_fields"] == ["amount_rub"]
                assert "amount_rub::ambiguous" in " ".join(kwargs["feedback"])
                data["fields"]["amount_rub"].update(status="ambiguous", value=None)
                data["fields"]["recipient_name"]["value"] = "Другое имя"
            return ExtractionProposal.model_validate(data)

    class Verifier(RejectNameOnce):
        async def ainvoke(self, state, questions, response_model):
            if hasattr(questions, "document_type"):
                return await super().ainvoke(state, questions, response_model)
            return response_model.model_validate({"answers": {
                key: {"noul": 0.95 if key == "amount_rub::ambiguous" else 0.01}
                for key in questions.root
            }})

    result = asyncio.run(process_document("local", dialogue, Verifier(), Extractor()))
    assert result.fields.amount_rub.status == "ambiguous"
    assert result.fields.amount_rub.value is None
    assert result.fields.recipient_name.value == "ИП Смирнов"


def test_missing_subject_question_checks_named_goods_without_price_rules():
    from src.field_rules import INVOICE_SUBJECT_RULE

    prepared = prepare_fields(invoice_proposal(), messages(), "invoice_draft")
    question = build_verification_questions(prepared).root["subject::status_wrong"].instructions
    assert INVOICE_SUBJECT_RULE in question
    assert "single billed goods/service category" in question
    assert "wrongly left missing" in question
    assert "several different" in question
    assert "total/count" not in question
    assert "unit_price_rub" not in question


def test_omitted_subject_feedback_repairs_only_subject_without_copying_item_attributes():
    dialogue = [DocumentMessage(id="m1", role="user", text="Счёт ИП Смирнову за монтаж на 1500 рублей")]

    class Extractor:
        calls = 0
        async def extract(self, *args, **kwargs):
            self.calls += 1
            data = invoice_proposal(name="ИП Смирнов").model_dump()
            if self.calls > 1:
                assert kwargs["repair_fields"] == ["subject"]
                data["fields"]["subject"] = {"status": "fillable", "value": "монтаж",
                    "evidence": [{"message_id": "m1", "quote": "монтаж"}]}
                # Посторонние изменения модели не должны проникнуть в результат.
                data["fields"]["amount_rub"]["value"] = 999
            return ExtractionProposal.model_validate(data)

    class SubjectVerifier(RejectNameOnce):
        async def ainvoke(self, state, questions, response_model):
            if hasattr(questions, "document_type"):
                return await super().ainvoke(state, questions, response_model)
            return response_model.model_validate({"answers": {
                key: {"noul": 0.9 if key == "subject::status_wrong" and
                      state.checked_fields.subject.status == "missing" else 0.01}
                for key in questions.root
            }})

    extractor = Extractor()
    result = asyncio.run(process_document("local", dialogue, SubjectVerifier(), extractor))
    assert extractor.calls == 2
    assert result.fields.subject.value == "монтаж"
    assert result.fields.amount_rub.value == 1500
