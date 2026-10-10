from typing import Annotated, Literal

from pydantic import BaseModel, Field
from typesafe_sdk import Usage

'''
class Ticket(Question):
    department: Choice = Field(..., alias="department")
    frustration: Score = Field(..., alias="frustration")
    is_urgent: Noul = Field(..., alias="is_urgent")

question = {
        "department": Choice(
            instructions="Which team should handle this",
            criteria={
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales": "Pricing or account questions",
            },
        ),
        "frustration": Score(
            instructions="How frustrated the customer appears",
            criteria=[
                "Calm, just stating facts",
                "Frustrated but civil",
                "Very angry, strong language",
            ],
        ),
        "is_urgent": Noul(
            instructions="The message conveys urgency or time-sensitivity",
        ),
    }
questions = Ticket(**question)
'''

type Probability = Annotated[float, Field(ge=0, le=1)]

type DocumentType = Literal[
    "invoice_draft",
    "transfer_draft",
    "none"
]

type NoneReasonType = Literal[
    "no_request",
    "other_document",
    "cancelled",
]

type DataSplit = Literal["dev", "test"]
type SplitFilter = Literal["dev", "test", "all"]

class Message(BaseModel):
    sender: Literal["customer", "support"]
    text: str


class Response[AnswersT: BaseModel](BaseModel):
    answers: AnswersT
    model: str
    usage: Usage
