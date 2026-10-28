from typing import Annotated, Literal

from pydantic import BaseModel, Field
from typesafe_sdk import Usage


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


class Response[AnswersT: BaseModel](BaseModel):
    answers: AnswersT
    model: str
    usage: Usage
