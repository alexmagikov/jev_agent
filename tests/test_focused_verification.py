"""Контракт запроса Jev: конкретный объект проверки, источники и полярность ошибки."""

import asyncio

from src.verification import verify_fields
from tests.test_verification import _inputs


def test_verification_sends_structured_claim_and_shared_context_without_duplicate_models():
    messages, proposal, prepared = _inputs()
    captured = []

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            captured.append((state.model_dump(mode="json"), questions.model_dump(mode="json")))
            return response_model.model_validate({"answers": {
                key: {"noul": 0.01} for key in questions.root
            }})

    assert asyncio.run(verify_fields(messages, "transfer_draft", proposal, prepared, Verifier())) == []
    state, questions = captured[0]
    assert set(state) == {"messages", "document_type", "rules"}
    assert state["messages"] == [message.model_dump(mode="json") for message in messages]
    assert "Assistant messages" in state["rules"]
    question = questions["amount_rub::unsupported"]
    claim = question["instructions"]["field"]
    assert claim["path"] == "amount_rub"
    assert claim["value"] == 100 and claim["status"] == "fillable"
    assert claim["meaning"] == "Сумма перевода в рублях"
    assert claim["evidence"] == [{"message_id": "m1", "quote": "100 рублей"}]
    assert "unsupported" in question["criteria"]["true"]
    assert "absent" in question["criteria"]["false"]
    assert all(state["rules"] not in str(q) for q in questions.values())


def test_structured_questions_pass_through_real_sdk_serializer():
    import json
    from scripts.measure_request_sizes import jev_body
    from src.verification import VerificationResponse, VerificationState, _RULES, build_verification_questions

    messages, proposal, prepared = _inputs()
    state = VerificationState(messages=messages, document_type="transfer_draft", rules=_RULES)
    questions = build_verification_questions(prepared, proposal=proposal)
    payload = json.loads(jev_body(state, questions, VerificationResponse))
    question = payload["questions"]["amount_rub::unsupported"]
    assert question["instructions"]["field"]["evidence"] == [{"message_id": "m1", "quote": "100 рублей"}]
    assert question["criteria"] == questions.root["amount_rub::unsupported"].criteria
    assert payload["state"] == state.model_dump(mode="json")
