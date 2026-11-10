"""LLM в роли модели решений: те же вопросы, что у Jev, ответ — вероятность ошибки; без API."""

import asyncio

import pytest

from src.cases.document_type import DocumentMessage
from src.llm import LLMResponseError
from src.llm_verifier import VERIFIER_PROMPT, LLMDecisionClient
from src.verification import NoulQuestion, VerificationQuestions, VerificationResponse, VerificationState

QUESTIONS = VerificationQuestions({
    "amount_rub::value_wrong": NoulQuestion(instructions={"question": "Is amount wrong?"},
                                            criteria={"true": "wrong", "false": "right"}),
    "items[0].description::noncanonical": NoulQuestion(instructions={"question": "Is it noncanonical?"},
                                                       criteria={"true": "yes", "false": "no"}),
})
STATE = VerificationState(messages=[DocumentMessage(id="m1", role="user", text="Счёт на 100 рублей")],
                          document_type="invoice_draft", rules="rules")


class FakeLLM:
    """Записывает запрос и возвращает подготовленные ответы вместо HTTP."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    async def complete_json(self, system, user, response_model, schema, schema_name):
        self.calls.append({"system": system, "user": user, "schema": schema, "name": schema_name})
        return response_model.model_validate({"answers": self.answers})


def ask(fake):
    return asyncio.run(LLMDecisionClient(fake, name="qwen").ainvoke(STATE, QUESTIONS, VerificationResponse))


def test_llm_gets_the_same_state_and_questions_as_jev_and_returns_probabilities():
    fake = FakeLLM([{"key": "amount_rub::value_wrong", "noul": 0.9},
                    {"key": "items[0].description::noncanonical", "noul": 0.1}])
    response = ask(fake)
    assert {key: answer.noul for key, answer in response.answers.items()} == {
        "amount_rub::value_wrong": 0.9, "items[0].description::noncanonical": 0.1}
    [call] = fake.calls
    assert call["system"] == VERIFIER_PROMPT
    assert call["user"]["state"] == STATE.model_dump(mode="json")
    assert call["user"]["questions"] == QUESTIONS.model_dump(mode="json")
    # Ключи с точками и скобками идут как значения enum, а не как имена свойств схемы.
    key_schema = call["schema"]["properties"]["answers"]["items"]["properties"]["key"]
    assert key_schema["enum"] == list(QUESTIONS.root)


@pytest.mark.parametrize("answers", [
    [{"key": "amount_rub::value_wrong", "noul": 0.9}],
    [{"key": "amount_rub::value_wrong", "noul": 0.9}, {"key": "amount_rub::value_wrong", "noul": 0.2},
     {"key": "items[0].description::noncanonical", "noul": 0.1}],
])
def test_missing_or_duplicate_answers_are_rejected(answers):
    with pytest.raises(LLMResponseError):
        ask(FakeLLM(answers))


def test_probability_outside_zero_one_is_rejected():
    with pytest.raises(LLMResponseError):
        ask(FakeLLM([{"key": "amount_rub::value_wrong", "noul": 1.5},
                     {"key": "items[0].description::noncanonical", "noul": 0.1}]))
