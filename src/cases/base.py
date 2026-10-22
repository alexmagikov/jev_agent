from pydantic import BaseModel

from src.client import DecisionModelClient


class Case[StateT: BaseModel, QuestionsT: BaseModel, ResponseT: BaseModel](BaseModel):
    """Связывает state, вопросы и схему ответа для одного запроса TypeSafe.

    Метод run передаёт данные клиенту и возвращает типизированный ответ.
    """

    state: StateT
    questions: QuestionsT
    response_model: type[ResponseT]

    async def run(self, client: DecisionModelClient) -> ResponseT:
        return await client.ainvoke(self.state, self.questions, self.response_model)
