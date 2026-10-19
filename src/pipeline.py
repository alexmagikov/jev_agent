
from collections.abc import Callable

from pydantic import BaseModel

from src.cases.document_type import (
    DocumentMessage,
    DocumentTypeCase,
    DocumentTypeState,
    NONE_REASON_TEXT,
    reason_text,
)
from src.client import AsyncJev, JevError, JevRateLimitError
from src.schemas.schemas import DocumentType, Probability
from src.schemas.document_schemas import (
    DocumentResult,
    InvoiceResult,
    NoDocumentResult,
    TransferResult,
)
from src.extraction import ExtractionError, ExtractionProposal, PreparedFields, prepare_fields, validate_messages
from src.llm import AsyncLLMExtractor, LLMError, LLMRateLimitError, LLMResponseError
from src.verification import VerificationError, verify_fields


class DocumentTypeDecision(BaseModel):
    """Результат первого этапа обработки диалога."""

    document_type: DocumentType
    confidence: Probability
    reason: str | None = None


async def classify_document(
    messages: list[DocumentMessage],
    client: AsyncJev,
) -> DocumentTypeDecision:
    """Определяет актуальный тип черновика и объясняет результат `none`.

    Оба вопроса (`document_type` и `none_reason`) отправляются в JEV одним
    запросом. Причина используется только тогда, когда JEV выбрал `none`.
    """

    case = DocumentTypeCase(state=DocumentTypeState(messages=messages))
    try:
        response = await case.run(client)
    except JevRateLimitError as error:
        raise PipelineRateLimitError(str(error)) from error
    except JevError as error:
        raise PipelineError(str(error)) from error

    document_type = response.answers.document_type
    none_reason = response.answers.none_reason

    return DocumentTypeDecision(
        document_type=document_type.choice,
        confidence=document_type.confidence,
        reason=reason_text(document_type, none_reason),
    )


class PipelineError(RuntimeError):
    """Кейс не прошёл проверки. Это не document_type=none и не missing."""


class PipelineRateLimitError(PipelineError):
    """Пакет нужно остановить, а не отправлять оставшиеся кейсы в ограниченный API."""


class PipelineRejectionError(PipelineError):
    """Попытки исчерпаны; диагностика содержит только внутренние коды, не ответ API."""

    def __init__(self, message: str, *, failure_reason: str, validation_failures: list[dict]):
        super().__init__(message)
        self.failure_reason = failure_reason
        self.validation_failures = validation_failures


async def process_document(
    case_id: str,
    messages: list[DocumentMessage],
    client: AsyncJev | None,
    extractor: AsyncLLMExtractor,
    *,
    max_attempts: int = 2,
    wrong_threshold: float = 0.7,
    debug: bool = False,
    use_jev: bool = True,
) -> DocumentResult:
    """С Jev: классификация -> LLM -> Python -> Jev; без: LLM -> LLM -> Python."""

    if not 1 <= max_attempts <= 3 or not 0 < wrong_threshold < 1:
        raise ValueError("Некорректные настройки числа попыток или порога")
    if use_jev and client is None:
        raise ValueError("Для режима с Jev нужен клиент")
    try:
        validate_messages(messages)
    except ExtractionError as error:
        raise PipelineError(f"{case_id}: {error}") from error

    if use_jev:
        if debug:
            print(f"[{case_id}] Классификация: запрос к Jev…", flush=True)
        decision = await classify_document(messages, client)
    else:
        classification_feedback: list[str] = []
        for attempt in range(1, max_attempts + 1):
            if debug:
                print(f"[{case_id}] Классификация LLM: попытка {attempt}/{max_attempts}…", flush=True)
            try:
                classification = await extractor.classify(messages, feedback=classification_feedback)
            except LLMResponseError as error:
                classification_feedback = [str(error)]
                if debug:
                    print(f"[{case_id}] Замечания к классификации: {error}", flush=True)
                continue
            except LLMRateLimitError as error:
                raise PipelineRateLimitError(f"{case_id}: {error}") from error
            except LLMError as error:
                raise PipelineError(f"{case_id}: {error}") from error
            
            decision = DocumentTypeDecision(
                document_type=classification.document_type,
                confidence=classification.confidence,
                reason=NONE_REASON_TEXT[classification.none_reason] if classification.none_reason else None,
            )
            break
        else:
            raise PipelineError(
                f"{case_id}: классификация не прошла за {max_attempts} попытки: "
                + "; ".join(classification_feedback)
            )
        
    return await process_selected_document(
        case_id, messages, decision, client, extractor,
        max_attempts=max_attempts, wrong_threshold=wrong_threshold, debug=debug, use_jev=use_jev,
    )


async def process_selected_document(
    case_id: str,
    messages: list[DocumentMessage],
    decision: DocumentTypeDecision,
    client: AsyncJev | None,
    extractor: AsyncLLMExtractor,
    *,
    max_attempts: int = 2,
    wrong_threshold: float = 0.7,
    debug: bool = False,
    use_jev: bool = True,
    initial_proposal: ExtractionProposal | None = None,
    on_candidate: Callable[[ExtractionProposal, PreparedFields], None] | None = None,
) -> DocumentResult:
    """Общий этап после классификации; initial_proposal позволяет парный аудит одного кандидата."""
    if not 1 <= max_attempts <= 3 or not 0 < wrong_threshold < 1:
        raise ValueError("Некорректные настройки числа попыток или порога")
    if use_jev and client is None:
        raise ValueError("Для режима с Jev нужен клиент")
    try:
        validate_messages(messages)
    except ExtractionError as error:
        raise PipelineError(f"{case_id}: {error}") from error
    if debug:
        print(f"[{case_id}] Тип: {decision.document_type}, confidence={decision.confidence:.3f}", flush=True)
    if decision.document_type == "none":
        return NoDocumentResult(
            case_id=case_id,
            document_type="none",
            confidence=decision.confidence,
            fields={},
            reason=decision.reason,
        )

    feedback: list[str] = []
    repair_base: ExtractionProposal | None = None
    repair_fields: list[str] = []
    validation_failures = []
    failure_reason = "semantic_rejection"

    for attempt in range(1, max_attempts + 1):
        if debug:
            print(f"[{case_id}] Попытка {attempt}/{max_attempts}: извлечение LLM…", flush=True)
        try:
            if attempt == 1 and initial_proposal is not None:
                proposal = initial_proposal.model_copy(deep=True)
            elif repair_base is None:
                proposal = await extractor.extract(messages, decision.document_type, feedback=feedback)
            else:
                proposal = await extractor.extract(
                    messages, decision.document_type, feedback=feedback,
                    previous_proposal=repair_base, repair_fields=repair_fields,
                )
                # Защита уже проверенных полей, даже если LLM проигнорировала область ремонта.
                proposal = ExtractionProposal(fields={
                    name: proposal.fields[name] if name in repair_fields else field
                    for name, field in repair_base.fields.items()
                })

            if debug:
                print(f"[{case_id}] Python: проверка источников, форматов и расчётов…", flush=True)

            fields = prepare_fields(proposal, messages, decision.document_type)
            if on_candidate is not None:
                on_candidate(proposal.model_copy(deep=True), fields.model_copy(deep=True))
        except (ExtractionError, LLMResponseError) as error:
            validation_failures.append({"attempt": attempt,
                                        "stage": "llm_response" if isinstance(error, LLMResponseError) else "python_validation",
                                        "code": error.code})
            failure_reason = "invalid_extraction"
            feedback = [str(error)]
            if debug:
                print(f"[{case_id}] Замечания LLM/Python: {feedback[0]}", flush=True)
            continue
        except LLMRateLimitError as error:
            raise PipelineRateLimitError(f"{case_id}: {error}") from error
        except LLMError as error:
            raise PipelineError(f"{case_id}: {error}") from error
        
        feedback = []
        failure_reason = "semantic_rejection"
        if use_jev:
            assert client is not None
            try:
                if debug:
                    print(f"[{case_id}] Jev: проверка полей…", flush=True)

                feedback = await verify_fields(
                    messages, 
                    decision.document_type,
                    proposal,
                    fields,
                    client,
                    wrong_threshold,
                    debug=debug,
                )
            except JevRateLimitError as error:
                raise PipelineRateLimitError(f"{case_id}: {error}") from error
            except (VerificationError, JevError) as error:
                raise PipelineError(f"{case_id}: {error}") from error

            if debug:
                print(f"[{case_id}] Замечания Jev:", flush=True)
                for remark in feedback:
                    print("  " + remark, flush=True)
                if not feedback:
                    print("  нет", flush=True)

        if not feedback:
            result_model = InvoiceResult if decision.document_type == "invoice_draft" else TransferResult
            return result_model(
                case_id=case_id, document_type=decision.document_type,
                confidence=decision.confidence, fields=fields,
            )
        
        repair_fields = sorted({remark.split("::", 1)[0].split("[", 1)[0] for remark in feedback})
        if not set(repair_fields) <= set(proposal.fields):
            raise PipelineError(f"{case_id}: проверка вернула неизвестное поле для исправления")
        repair_base = proposal.model_copy(deep=True)

    summary = "; ".join(feedback)
    raise PipelineRejectionError(
        f"{case_id}: проверки не пройдены за {max_attempts} попытки: {summary}",
        failure_reason=failure_reason, validation_failures=validation_failures,
    )
