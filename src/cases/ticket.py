from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel
from typesafe_sdk import Noul, NoulAnswer, Score

from src.cases.base import Case
from src.schemas.schemas import Probability, Response


class TicketState(RootModel[str]):
    pass


class Departments[ValueT](BaseModel):
    billing: ValueT
    technical: ValueT
    sales: ValueT


class DepartmentQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str = "Which team should handle this message?"
    criteria: Departments[str] = Departments[str](
        billing="Payment or subscription issues",
        technical="Bugs or integration problems",
        sales="Pricing or account questions",
    )


class TicketQuestions(BaseModel):
    department: DepartmentQuestion = DepartmentQuestion()
    frustration: Score = Score(
        instructions="How frustrated does the customer appear?",
        criteria=(
            "Calm, just stating facts",
            "Frustrated but civil",
            "Very angry, strong language",
        ),
    )
    is_urgent: Noul = Noul(
        instructions="The message conveys urgency or time-sensitivity",
    )


class DepartmentAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: Literal["billing", "technical", "sales"]
    confidence: Probability
    probabilities: Departments[Probability]


class FrustrationLevels[ValueT](BaseModel):
    model_config = ConfigDict(validate_by_name=True)

    calm: ValueT = Field(alias="0")
    frustrated: ValueT = Field(alias="1")
    angry: ValueT = Field(alias="2")


class FrustrationAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float = Field(ge=0, le=2)
    confidence: Probability
    legend: FrustrationLevels[str]
    probabilities: FrustrationLevels[Probability]


class TicketAnswers(BaseModel):
    department: DepartmentAnswer
    frustration: FrustrationAnswer
    is_urgent: NoulAnswer


TicketResponse = Response[TicketAnswers]


class TicketCase(Case[TicketState, TicketQuestions, TicketResponse]):
    """Классифицирует текст обращения в поддержку.

    State — строка с сообщением клиента. Вопросы определяют отдел (Choice),
    степень недовольства (Score от 0 до 2) и вероятность срочности (Noul).
    Результат — TicketResponse с ответами и метаданными запроса.
    """

    state: TicketState = TicketState(
        "Hi, I've been trying to connect my Stripe account for 3 days "
        "and the integration keeps failing. I'm losing sales. Please help ASAP."
    )
    questions: TicketQuestions = TicketQuestions()
    response_model: type[TicketResponse] = TicketResponse
