
from collections.abc import Callable

from pydantic import BaseModel

from src.cases.document_type import (
    DocumentMessage,
    DocumentTypeCase,
    DocumentTypeState,
    NONE_REASON_TEXT,
    reason_text,
)
from src.client import DecisionModelClient, DecisionModelError, DecisionModelRateLimitError
from src.schemas.schemas import DocumentType, Probability
from src.schemas.document_schemas import (
    DocumentResult,
    DocumentValidation,
    ValidationIssue,
    InvoiceResult,
    NoDocumentResult,
    TransferResult,
)
from src.extraction import ExtractionError, ExtractionProposal, PreparedFields, prepare_fields, prepare_partial_fields, validate_messages
from src.llm import AsyncLLMExtractor, LLMError, LLMRateLimitError, LLMResponseError
from src.verification import VerificationError, verify_fields
from src.repairs import merge_repair, plan_repair, repair_roots


class DocumentTypeDecision(BaseModel):
    """Результат первого этапа обработки диалога."""

    document_type: DocumentType
    confidence: Probability
    reason: str | None = None


async def classify_document(
    messages: list[DocumentMessage],
    client: DecisionModelClient,
) -> DocumentTypeDecision:
    """Определяет актуальный тип черновика и объясняет результат `none`.

    Оба вопроса (`document_type` и `none_reason`) отправляются модели одним
    запросом. Причина используется только тогда, когда JEV выбрал `none`.
    """

    case = DocumentTypeCase(state=DocumentTypeState(messages=messages))
    try:
        response = await case.run(client)
    except DecisionModelRateLimitError as error:
        raise PipelineRateLimitError(str(error)) from error
    except DecisionModelError as error:
        raise PipelineError(str(error)) from error

    document_type = response.answers.document_type
    none_reason = response.answers.none_reason

    return DocumentTypeDecision(
        document_type=document_type.choice,
        confidence=document_type.confidence,
        reason=reason_text(document_type, none_reason),
    )


class PipelineError(RuntimeError):
    """Не удалось получить документ: ошибка входа, сервиса или структуры ответа."""


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
    client: DecisionModelClient | None,
    extractor: AsyncLLMExtractor,
    *,
    max_attempts: int = 2,
    wrong_threshold: float = 0.7,
    debug: bool = False,
    use_verifier: bool = True,
    allow_partial: bool = True,
) -> DocumentResult:
    """Внешняя модель: классификация -> LLM -> Python -> проверка; без: LLM -> LLM -> Python.

    use_verifier включает внешнюю классификацию и смысловую проверку полей.
    """

    if not 1 <= max_attempts <= 3 or not 0 < wrong_threshold < 1:
        raise ValueError("Некорректные настройки числа попыток или порога")
    if use_verifier and client is None:
        raise ValueError("Для режима с верификатором нужен клиент")
    try:
        validate_messages(messages)
    except ExtractionError as error:
        raise PipelineError(f"{case_id}: {error}") from error

    if use_verifier:
        if debug:
            print(f"[{case_id}] Классификация: запрос к {getattr(client, 'name', 'Decision model')}…", flush=True)
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
        max_attempts=max_attempts, wrong_threshold=wrong_threshold, debug=debug, use_verifier=use_verifier,
        allow_partial=allow_partial,
    )


async def process_selected_document(
    case_id: str,
    messages: list[DocumentMessage],
    decision: DocumentTypeDecision,
    client: DecisionModelClient | None,
    extractor: AsyncLLMExtractor,
    *,
    max_attempts: int = 2,
    wrong_threshold: float = 0.7,
    debug: bool = False,
    use_verifier: bool = True,
    allow_partial: bool = True,
    initial_proposal: ExtractionProposal | None = None,
    on_candidate: Callable[[ExtractionProposal, PreparedFields], None] | None = None,
) -> DocumentResult:
    """Извлекает и ремонтирует документ; после исчерпания попыток сохраняет замечания.

    allow_partial=False оставляет строгий отказ для сравнительных аудитов.
    initial_proposal позволяет повторно проверить одного и того же кандидата.
    """
    if not 1 <= max_attempts <= 3 or not 0 < wrong_threshold < 1:
        raise ValueError("Некорректные настройки числа попыток или порога")
    if use_verifier and client is None:
        raise ValueError("Для режима с верификатором нужен клиент")
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
    repair_checked_fields: PreparedFields | None = None
    repair_fields: list[str] = []
    repair_paths: list[str] = []
    validation_failures = []
    failure_reason = "semantic_rejection"
    latest_proposal: ExtractionProposal | None = None
    latest_fields: PreparedFields | None = None
    latest_semantic_feedback: list[str] = []
    response_issues: list[ValidationIssue] = []

    for attempt in range(1, max_attempts + 1):
        if debug:
            print(f"[{case_id}] Попытка {attempt}/{max_attempts}: извлечение LLM…", flush=True)
        proposal = None
        try:
            if attempt == 1 and initial_proposal is not None:
                proposal = initial_proposal.model_copy(deep=True)
            elif repair_base is None:
                proposal = await extractor.extract(messages, decision.document_type, feedback=feedback)
            else:
                reply = await extractor.extract(
                    messages, decision.document_type, feedback=feedback,
                    previous_proposal=repair_base, repair_fields=repair_fields, repair_paths=repair_paths,
                    previous_checked_fields=repair_checked_fields,
                )
                # Защита уже проверенных полей, даже если LLM проигнорировала область ремонта.
                proposal = merge_repair(repair_base, reply, repair_paths)

            if debug:
                print(f"[{case_id}] Python: проверка источников, форматов и расчётов…", flush=True)

            latest_proposal = proposal.model_copy(deep=True)
            latest_fields = None
            latest_semantic_feedback = []
            response_issues = []
            fields = prepare_fields(proposal, messages, decision.document_type)
            latest_fields = fields.model_copy(deep=True)
            if on_candidate is not None:
                on_candidate(proposal.model_copy(deep=True), fields.model_copy(deep=True))
        except (ExtractionError, LLMResponseError) as error:
            validation_failures.append({"attempt": attempt,
                                        "stage": "llm_response" if isinstance(error, LLMResponseError) else "python_validation",
                                        "code": error.code})
            
            failure_reason = "invalid_extraction"
            feedback = [str(error)]
            if isinstance(error, LLMResponseError):
                response_issues.append(ValidationIssue(
                    path="", stage="llm_response", code=error.code,
                    message="Повторный ответ LLM не удалось применить. Сохранён предыдущий кандидат.",
                ))

            if isinstance(error, ExtractionError) and proposal is not None:
                repair_checked_fields = None  # Этот кандидат ещё не прошёл Python целиком.
                if error.repair_paths:
                    repair_paths = plan_repair(proposal, list(error.repair_paths))
                    repair_fields = repair_roots(repair_paths)
                    repair_base = proposal.model_copy(deep=True)
                else:
                    # Без надёжного пути безопасного точечного ремонта нет.
                    repair_base = None
                    repair_paths, repair_fields = [], []

            if debug:
                print(f"[{case_id}] Замечания LLM/Python: {feedback[0]}", flush=True)
            continue
        except LLMRateLimitError as error:
            raise PipelineRateLimitError(f"{case_id}: {error}") from error
        except LLMError as error:
            raise PipelineError(f"{case_id}: {error}") from error
        
        feedback = []
        failure_reason = "semantic_rejection"
        if use_verifier:
            assert client is not None
            try:
                if debug:
                    print(f"[{case_id}] {getattr(client, 'name', 'Decision model')}: проверка полей…", flush=True)

                feedback = await verify_fields(
                    messages, 
                    decision.document_type,
                    proposal,
                    fields,
                    client,
                    wrong_threshold,
                    debug=debug,
                )
            except DecisionModelRateLimitError as error:
                raise PipelineRateLimitError(f"{case_id}: {error}") from error
            except (VerificationError, DecisionModelError) as error:
                raise PipelineError(f"{case_id}: {error}") from error

            latest_semantic_feedback = feedback.copy()
            if debug:
                print(f"[{case_id}] Замечания {getattr(client, 'name', 'Decision model')}:", flush=True)
                for remark in feedback:
                    print("  " + remark, flush=True)
                if not feedback:
                    print("  нет", flush=True)

        if not feedback:
            result_model = InvoiceResult if decision.document_type == "invoice_draft" else TransferResult
            return result_model(
                case_id=case_id, document_type=decision.document_type,
                confidence=decision.confidence, fields=fields,
                validation=DocumentValidation(semantic_status="passed" if use_verifier else "not_run"),
            )
        
        try:
            repair_paths = plan_repair(proposal, [remark.split("::", 1)[0] for remark in feedback])
        except ValueError as error:
            raise PipelineError(f"{case_id}: проверка вернула неизвестное поле для исправления") from error
        repair_fields = repair_roots(repair_paths)
        repair_base = proposal.model_copy(deep=True)
        repair_checked_fields = fields.model_copy(deep=True)

    if allow_partial and latest_proposal is not None:
        if latest_fields is None:
            try:
                fields, validation = prepare_partial_fields(latest_proposal, messages, decision.document_type)
            except ExtractionError:
                # Несовместимая структура шаблона не является локальной ошибкой поля.
                fields = None
        else:
            fields = latest_fields.model_copy(deep=True)
            validation = DocumentValidation(semantic_status="needs_review" if latest_semantic_feedback else "not_run")
        if fields is not None:
            for remark in latest_semantic_feedback:
                key = remark.split(": ", 1)[0]
                path, code = key.split("::", 1)
                validation.issues.append(ValidationIssue(
                    path=path, stage="semantic", code=code, message=remark,
                ))
                name = repair_roots([path])[0]
                getattr(fields, name).validation_status = "needs_review"
                validation.unverified_fields[name] = latest_proposal.fields[name].model_dump(mode="json")
            validation.issues.extend(response_issues)
            if validation.issues:
                validation.status = "needs_review"
            result_model = InvoiceResult if decision.document_type == "invoice_draft" else TransferResult
            return result_model(
                case_id=case_id, document_type=decision.document_type,
                confidence=decision.confidence, fields=fields, validation=validation,
            )

    summary = "; ".join(feedback)
    raise PipelineRejectionError(
        f"{case_id}: проверки не пройдены за {max_attempts} попытки: {summary}",
        failure_reason=failure_reason, validation_failures=validation_failures,
    )
