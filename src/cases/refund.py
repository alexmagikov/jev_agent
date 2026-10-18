from typing import Literal

from pydantic import BaseModel
from typesafe_sdk import Noul, NoulAnswer

from src.cases.base import Case
from src.schemas.schemas import Message, Response


class Charge(BaseModel):
    amount_usd: float
    status: Literal["captured", "refunded"]


class Order(BaseModel):
    id: str
    charges: list[Charge]


class RefundState(BaseModel):
    messages: list[Message]
    order: Order
    refund_policy: str


class PolicyInstructions(BaseModel):
    question: str = "Does the policy support the customer's refund request?"
    compare: tuple[str, ...] = ("messages", "order.charges", "refund_policy")
    focus: str = "Evaluate the customer's request against the charges and the refund policy."


class PolicyQuestion(BaseModel):
    type: Literal["noul"] = "noul"
    instructions: PolicyInstructions = PolicyInstructions()


class RefundQuestions(BaseModel):
    refund_requested: Noul = Noul(
        instructions="Do the customer's messages in `messages` request a refund?",
    )
    duplicate_charge: Noul = Noul(
        instructions="Does `order.charges` contain a duplicate captured charge for `order.id`?",
    )
    policy_supports_refund: PolicyQuestion = PolicyQuestion()


class RefundAnswers(BaseModel):
    refund_requested: NoulAnswer
    duplicate_charge: NoulAnswer
    policy_supports_refund: NoulAnswer


RefundResponse = Response[RefundAnswers]


class RefundCase(Case[RefundState, RefundQuestions, RefundResponse]):
    """Оценивает запрос возврата с учётом заказа и политики возврата.

    State — объект с перепиской, списаниями и правилами возврата. Три Noul
    оценивают вероятность запроса возврата, повторного списания и соответствия
    запроса политике. Результат — RefundResponse; решение о возврате принимает код.
    """

    state: RefundState = RefundState(
        messages=[
            Message(sender="customer", text="I was charged twice for A-104. Please refund the duplicate."),
            Message(sender="support", text="We are checking the charges."),
        ],
        order=Order(
            id="A-104",
            charges=[
                Charge(amount_usd=49, status="captured"),
                Charge(amount_usd=49, status="captured"),
            ],
        ),
        refund_policy="Duplicate captured charges are eligible for a refund.",
    )
    questions: RefundQuestions = RefundQuestions()
    response_model: type[RefundResponse] = RefundResponse
