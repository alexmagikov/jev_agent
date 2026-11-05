"""LLM в роли модели решений: базовая линия для сравнения с Jev.

Получает те же state и noul-вопросы, что Jev, и отвечает вероятностью ошибки на каждый.
Транспорт, JSON-схема и учёт расхода — общие с извлекающей LLM (AsyncLLMExtractor).
"""

from pydantic import BaseModel, ValidationError

from src.llm import LLMResponseError

VERIFIER_PROMPT = """You are an independent verifier of a draft bank document extracted from a Russian dialogue.
The input has state (dialogue messages, document type, rules) and questions. Each question describes one
specific possible error in one field of the draft: see its instructions (including the field) and criteria.
For every question, estimate the probability from 0 to 1 that the described error IS present, that is,
criteria.true holds. Judge only that question; other errors do not count.
Only user messages are evidence; assistant messages are not user data.
Be calibrated: close to 1 only when the error is clearly present, close to 0 when the draft is clearly right,
intermediate values when unsure.
Return JSON only: {"answers": [{"key": <question key>, "noul": <probability>}]}, exactly one answer per key."""


class _KeyedAnswer(BaseModel):
    key: str
    noul: float


class _LLMVerdict(BaseModel):
    answers: list[_KeyedAnswer]


def _schema(keys: list[str]) -> dict:
    # Ключи вида items[0].description::noncanonical — значения enum, а не имена свойств:
    # так схема остаётся допустимой для строгого json_schema у провайдеров.
    return {
        "type": "object", "additionalProperties": False, "required": ["answers"],
        "properties": {"answers": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["key", "noul"],
            "properties": {"key": {"type": "string", "enum": keys}, "noul": {"type": "number"}},
        }}},
    }


class LLMDecisionClient:
    """Совместим с DecisionModelClient.ainvoke для noul-вопросов проверки полей."""

    name = "LLM verifier"

    def __init__(self, llm, *, name: str | None = None):
        self.llm = llm
        if name is not None:
            self.name = name

    async def ainvoke(self, state: BaseModel, questions: BaseModel, response_model):
        keys = list(questions.root)
        user = {"state": state.model_dump(mode="json"), "questions": questions.model_dump(mode="json")}
        verdict = await self.llm.complete_json(VERIFIER_PROMPT, user, _LLMVerdict, _schema(keys), "field_verification")
        answers = {}
        for answer in verdict.answers:
            if answer.key in answers:
                raise LLMResponseError(f"{self.name}: повторный ответ на вопрос")
            answers[answer.key] = {"noul": answer.noul}
        if set(answers) != set(keys):
            raise LLMResponseError(f"{self.name}: ответ не на все вопросы или на лишние")
        try:
            return response_model.model_validate({"answers": answers})
        except ValidationError:
            raise LLMResponseError(f"{self.name}: вероятность вне диапазона 0–1") from None
