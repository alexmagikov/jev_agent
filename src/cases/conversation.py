from pydantic import BaseModel, RootModel
from typesafe_sdk import Noul, NoulAnswer

from src.cases.base import Case
from src.schemas.schemas import Message, Response


class ConversationState(RootModel[list[Message]]):
    pass


class ConversationQuestions(BaseModel):
    human_requested: Noul = Noul(
        instructions="Do the customer's messages ask to speak to a human agent?",
    )
    repeat_contact: Noul = Noul(
        instructions="Do the customer's messages indicate previous contact about this issue?",
    )


class ConversationAnswers(BaseModel):
    human_requested: NoulAnswer
    repeat_contact: NoulAnswer


ConversationResponse = Response[ConversationAnswers]


class ConversationCase(Case[ConversationState, ConversationQuestions, ConversationResponse]):
    """Анализирует переписку клиента с поддержкой.

    State — массив сообщений с отправителем и текстом. Два Noul оценивают
    вероятность запроса живого оператора и повторного обращения по той же проблеме.
    Результат — ConversationResponse с ответами и метаданными запроса.
    """

    state: ConversationState = ConversationState([
        Message(sender="customer", text="This is the third time I have contacted you."),
        Message(sender="support", text="Could you describe the issue?"),
        Message(sender="customer", text="Can I please speak to a real person?"),
    ])
    questions: ConversationQuestions = ConversationQuestions()
    response_model: type[ConversationResponse] = ConversationResponse
