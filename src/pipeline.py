
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

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
    NoDocumentResult,
    RESULT_BY_TYPE,
    field_root,
)
from src.extraction import (
    ExtractionError, ExtractionProposal, PreparedFields, amount_depends_on_unverified_items, dependent_amount_issue,
    prepare_fields, prepare_partial_fields, validate_messages,
)
from src.llm import AsyncLLMExtractor, LLMError, LLMRateLimitError, LLMResponseError
from src.verification import Remark, VerificationError, verify_fields
from src.repairs import merge_repair, plan_repair, repair_roots


logger = logging.getLogger(__name__)


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
    wrong_threshold: float = 0.8,
    use_verifier: bool = True,
    allow_partial: bool = True,
) -> DocumentResult:
    """Внешняя модель: классификация -> LLM -> Python -> проверка; без: LLM -> LLM -> Python.

    use_verifier включает внешнюю классификацию и смысловую проверку полей.
    """
    _check_inputs(case_id, messages, client, max_attempts=max_attempts,
                  wrong_threshold=wrong_threshold, use_verifier=use_verifier)

    if use_verifier:
        logger.debug("[%s] Классификация: запрос к %s…", case_id, getattr(client, "name", "Decision model"))
        decision = await classify_document(messages, client)
    else:
        classification_feedback: list[str] = []
        for attempt in range(1, max_attempts + 1):
            logger.debug("[%s] Классификация LLM: попытка %d/%d…", case_id, attempt, max_attempts)
            try:
                classification = await extractor.classify(messages, feedback=classification_feedback)
            except LLMResponseError as error:
                classification_feedback = [str(error)]
                logger.debug("[%s] Замечания к классификации: %s", case_id, error)
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

    return await _extract_and_verify(
        case_id, messages, decision, client, extractor,
        max_attempts=max_attempts, wrong_threshold=wrong_threshold, use_verifier=use_verifier,
        allow_partial=allow_partial,
    )


def _check_inputs(
    case_id: str, messages: list[DocumentMessage], client: DecisionModelClient | None, *,
    max_attempts: int, wrong_threshold: float, use_verifier: bool,
) -> None:
    """Проверка входа на публичных точках пайплайна, до платных запросов."""
    if not 1 <= max_attempts <= 3 or not 0 < wrong_threshold < 1:
        raise ValueError("Некорректные настройки числа попыток или порога")
    if use_verifier and client is None:
        raise ValueError("Для режима с верификатором нужен клиент")
    try:
        validate_messages(messages)
    except ExtractionError as error:
        raise PipelineError(f"{case_id}: {error}") from error


async def process_selected_document(
    case_id: str,
    messages: list[DocumentMessage],
    decision: DocumentTypeDecision,
    client: DecisionModelClient | None,
    extractor: AsyncLLMExtractor,
    *,
    max_attempts: int = 2,
    wrong_threshold: float = 0.8,
    use_verifier: bool = True,
    allow_partial: bool = True,
    initial_proposal: ExtractionProposal | None = None,
    on_candidate: Callable[[ExtractionProposal, PreparedFields], None] | None = None,
) -> DocumentResult:
    """Извлекает и ремонтирует документ уже выбранного типа; после исчерпания попыток сохраняет замечания.

    Из всех кандидатов сохраняется лучший: прошедший Python, затем с меньшим
    числом непроверенных полей, при равенстве — более поздний.
    allow_partial=False оставляет строгий отказ для сравнительных аудитов.
    initial_proposal позволяет повторно проверить одного и того же кандидата.
    """
    _check_inputs(case_id, messages, client, max_attempts=max_attempts,
                  wrong_threshold=wrong_threshold, use_verifier=use_verifier)
    return await _extract_and_verify(
        case_id, messages, decision, client, extractor,
        max_attempts=max_attempts, wrong_threshold=wrong_threshold, use_verifier=use_verifier,
        allow_partial=allow_partial, initial_proposal=initial_proposal, on_candidate=on_candidate,
    )


@dataclass
class _RepairPlan:
    """Что чинить на следующей попытке: базовый кандидат, его проверенные поля и пути ремонта."""

    base: ExtractionProposal | None = None
    checked_fields: PreparedFields | None = None
    roots: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)

    def after_python_error(self, proposal: ExtractionProposal, error: ExtractionError) -> None:
        self.checked_fields = None  # Этот кандидат ещё не прошёл Python целиком.
        if error.repair_paths:
            self.paths = plan_repair(proposal, list(error.repair_paths))
            self.roots = repair_roots(self.paths)
            self.base = proposal.model_copy(deep=True)
        else:
            # Без надёжного пути безопасного точечного ремонта нет.
            self.base = None
            self.paths, self.roots = [], []

    def after_remarks(self, proposal: ExtractionProposal, fields: PreparedFields, remarks: list[Remark]) -> None:
        self.paths = plan_repair(proposal, [remark.path for remark in remarks])
        self.roots = repair_roots(self.paths)
        self.base = proposal.model_copy(deep=True)
        self.checked_fields = fields.model_copy(deep=True)


async def _extract_and_verify(
    case_id: str,
    messages: list[DocumentMessage],
    decision: DocumentTypeDecision,
    client: DecisionModelClient | None,
    extractor: AsyncLLMExtractor,
    *,
    max_attempts: int,
    wrong_threshold: float,
    use_verifier: bool,
    allow_partial: bool,
    initial_proposal: ExtractionProposal | None = None,
    on_candidate: Callable[[ExtractionProposal, PreparedFields], None] | None = None,
) -> DocumentResult:
    """Попытки извлечения и ремонта; вход уже проверен _check_inputs."""
    logger.debug("[%s] Тип: %s, confidence=%.3f", case_id, decision.document_type, decision.confidence)
    if decision.document_type == "none":
        return NoDocumentResult(
            case_id=case_id,
            document_type="none",
            confidence=decision.confidence,
            fields={},
            reason=decision.reason,
            validation=DocumentValidation(),
        )

    feedback: list[str] = []
    plan = _RepairPlan()
    validation_failures = []
    failure_reason = "semantic_rejection"
    # Кандидат, его поля после Python (None — не прошёл) и замечания верификатора.
    candidates: list[tuple[ExtractionProposal, PreparedFields | None, list[Remark]]] = []
    response_issues: list[ValidationIssue] = []

    for attempt in range(1, max_attempts + 1):
        logger.debug("[%s] Попытка %d/%d: извлечение LLM…", case_id, attempt, max_attempts)
        proposal = None
        try:
            if attempt == 1 and initial_proposal is not None:
                proposal = initial_proposal.model_copy(deep=True)
            else:
                proposal = await _next_candidate(extractor, messages, decision.document_type, feedback, plan)
            logger.debug("[%s] Python: проверка источников, форматов и расчётов…", case_id)
            response_issues = []
            fields = prepare_fields(proposal, messages, decision.document_type)
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
                    message="Повторный ответ LLM не удалось применить. Сохранён лучший из прежних кандидатов.",
                ))
            if isinstance(error, ExtractionError) and proposal is not None:
                candidates.append((proposal.model_copy(deep=True), None, []))
                plan.after_python_error(proposal, error)
            logger.debug("[%s] Замечания LLM/Python: %s", case_id, feedback[0])
            continue
        except LLMRateLimitError as error:
            raise PipelineRateLimitError(f"{case_id}: {error}") from error
        except LLMError as error:
            raise PipelineError(f"{case_id}: {error}") from error

        failure_reason = "semantic_rejection"
        remarks: list[Remark] = []
        if use_verifier:
            assert client is not None
            remarks = await _verify(case_id, messages, decision.document_type, proposal, fields, client, wrong_threshold)
        feedback = [str(remark) for remark in remarks]
        if not remarks:
            return _document_result(case_id, decision, fields, DocumentValidation(
                semantic_status="passed" if use_verifier else "not_run"))

        candidates.append((proposal.model_copy(deep=True), fields.model_copy(deep=True), list(remarks)))
        try:
            plan.after_remarks(proposal, fields, remarks)
        except ValueError as error:
            raise PipelineError(f"{case_id}: проверка вернула неизвестное поле для исправления") from error

    if allow_partial:
        best = _best_candidate(candidates, messages, decision.document_type)
        if best is not None:
            fields, validation = best
            validation.issues.extend(response_issues)
            if validation.issues:
                validation.status = "needs_review"
            return _document_result(case_id, decision, fields, validation)

    summary = "; ".join(feedback)
    raise PipelineRejectionError(
        f"{case_id}: проверки не пройдены за {max_attempts} попытки: {summary}",
        failure_reason=failure_reason, validation_failures=validation_failures,
    )


async def _next_candidate(
    extractor: AsyncLLMExtractor, messages: list[DocumentMessage], document_type: str,
    feedback: list[str], plan: _RepairPlan,
) -> ExtractionProposal:
    """Новое извлечение или ремонт по плану; проверенные поля вне плана не меняются."""
    if plan.base is None:
        return await extractor.extract(messages, document_type, feedback=feedback)
    reply = await extractor.extract(
        messages, document_type, feedback=feedback,
        previous_proposal=plan.base, repair_fields=plan.roots, repair_paths=plan.paths,
        previous_checked_fields=plan.checked_fields,
    )
    # Защита уже проверенных полей, даже если LLM проигнорировала область ремонта.
    return merge_repair(plan.base, reply, plan.paths)


async def _verify(
    case_id: str, messages: list[DocumentMessage], document_type: str, proposal: ExtractionProposal,
    fields: PreparedFields, client: DecisionModelClient, wrong_threshold: float,
) -> list[Remark]:
    """Смысловая проверка Jev; ошибки сервиса превращаются в ошибки пайплайна."""
    verifier = getattr(client, "name", "Decision model")
    try:
        logger.debug("[%s] %s: проверка полей…", case_id, verifier)
        remarks = await verify_fields(messages, document_type, proposal, fields, client, wrong_threshold)
    except DecisionModelRateLimitError as error:
        raise PipelineRateLimitError(f"{case_id}: {error}") from error
    except (VerificationError, DecisionModelError) as error:
        raise PipelineError(f"{case_id}: {error}") from error
    logger.debug("[%s] Замечания %s:%s", case_id, verifier,
                 "".join(f"\n  {remark}" for remark in remarks) or " нет")
    return remarks


def _best_candidate(
    candidates: list[tuple[ExtractionProposal, PreparedFields | None, list[Remark]]],
    messages: list[DocumentMessage], document_type: str,
) -> tuple[PreparedFields, DocumentValidation] | None:
    """Лучший кандидат для needs_review: прошедший Python, с меньшим числом
    непроверенных полей, при равенстве — более поздний."""
    best = None
    for index, (proposal, checked_fields, remarks) in enumerate(candidates):
        reviewed = _review_candidate(proposal, checked_fields, remarks, messages, document_type)
        if reviewed is not None:
            rank = (checked_fields is not None, -len(reviewed[1].unverified_fields), index)
            if best is None or rank > best[0]:
                best = (rank, reviewed)
    return best[1] if best is not None else None


def _document_result(
    case_id: str, decision: DocumentTypeDecision, fields: PreparedFields, validation: DocumentValidation,
) -> DocumentResult:
    return RESULT_BY_TYPE[decision.document_type](
        case_id=case_id, document_type=decision.document_type,
        confidence=decision.confidence, fields=fields, validation=validation,
    )


def _review_candidate(
    proposal: ExtractionProposal,
    checked_fields: PreparedFields | None,
    remarks: list[Remark],
    messages: list[DocumentMessage],
    document_type: str,
) -> tuple[PreparedFields, DocumentValidation] | None:
    """Поля и замечания сохраняемого кандидата; None — несовместимая структура шаблона."""
    if checked_fields is None:
        try:
            return prepare_partial_fields(proposal, messages, document_type)
        except ExtractionError:
            # Несовместимая структура шаблона не является локальной ошибкой поля.
            return None
    fields = checked_fields.model_copy(deep=True)
    validation = DocumentValidation(semantic_status="needs_review" if remarks else "not_run")
    for remark in remarks:
        validation.issues.append(ValidationIssue(
            path=remark.path, stage="semantic", code=remark.check, message=str(remark),
        ))
        name = field_root(remark.path)
        getattr(fields, name).validation_status = "needs_review"
        validation.unverified_fields[name] = proposal.fields[name].model_dump(mode="json")
    if amount_depends_on_unverified_items(proposal, set(validation.unverified_fields)):
        validation.issues.append(dependent_amount_issue("semantic"))
        fields.amount_rub.validation_status = "needs_review"
        validation.unverified_fields["amount_rub"] = proposal.fields["amount_rub"].model_dump(mode="json")
    return fields, validation
