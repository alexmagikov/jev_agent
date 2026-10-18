import asyncio
from types import SimpleNamespace

from src.cases.document_type import DocumentMessage
from src.schemas.document_schemas import TransferFields
from src.extraction import ExtractionProposal
from src.pipeline import process_document


class LowConfidenceJev:
    async def ainvoke(self, state, questions, response_model):
        if hasattr(questions, "document_type"):
            return SimpleNamespace(answers=SimpleNamespace(
                document_type=SimpleNamespace(choice="transfer_draft", confidence=0.01),
                none_reason=SimpleNamespace(choice="no_request"),
            ))
        answers = {key: {"noul": 0.01} for key in questions.root}
        return response_model.model_validate({"answers": answers})


class TransferExtractor:
    async def extract(self, messages, document_type, feedback=None):
        fields = {
            name: {"status": "missing", "value": None, "evidence": []}
            for name in TransferFields.model_fields
        }
        fields["amount_rub"] = {
            "status": "fillable",
            "value": 100,
            "evidence": [{"message_id": "m1", "quote": "100 рублей"}],
        }
        return ExtractionProposal.model_validate({"fields": fields})


def test_low_classification_confidence_does_not_block_current_pipeline():
    messages = [DocumentMessage(id="m1", role="user", text="Переведи себе 100 рублей")]
    result = asyncio.run(process_document("C1", messages, LowConfidenceJev(), TransferExtractor()))
    assert result.document_type == "transfer_draft"
    assert result.confidence == 0.01
    assert result.fields.amount_rub.value == 100
