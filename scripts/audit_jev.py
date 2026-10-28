"""Парный аудит Jeff/CLM/Jev/Laya: общий ответ LLM, реальные исправления, неизменные метрики."""

import argparse
import asyncio
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError
from typesafe_sdk import AsyncTypeSafeClient, TypeSafeError, TypeSafeAPIError, TypeSafeAPIConnectionError

from src.cases.document_type import NONE_REASON_TEXT
from src.client import DecisionUsage, decision_sdk_options, clm_sdk_options, jeff_sdk_options, laya_sdk_options, laya_request_options, prepare_clm_questions
from src.extraction import ExtractionProposal, prepare_fields
from src.detection_metrics import calculate_detection_metrics, calculate_field_outcomes, print_detection_metrics, print_field_outcomes
from src.llm import LLMUsage, LLMError, LLMResponseError, LLMRateLimitError, build_extraction_prompt
from src.repairs import merge_repair
from src.request_budget import APICallBudget, APICallBudgetError as AuditBudgetError, BudgetedExtractor
from src.schemas.document_schemas import InvoiceFields, TransferFields
from src.main import DEFAULT_INPUT_PATH, DEFAULT_LABELS_PATH, estimate_usage_costs, load_cases, load_labels, print_acceptance_metrics
from src.metrics import AcceptanceMetrics, calculate_metrics, field_matches
from src.pipeline import DocumentTypeDecision, PipelineError, PipelineRejectionError, process_selected_document
from src.settings import CLMSettings, JeffSettings, DecisionModelSettings, LayaSettings, LLMSettings, PipelineSettings


class AuditResumeError(ValueError):
    """Безопасное объяснение несовместимости трассы, без ответов API и реквизитов."""


def usage_snapshot(client):
    usage = getattr(client, "usage", None)
    if not is_dataclass(usage):
        return None
    result = asdict(usage)
    budget = getattr(client, "budget", None)
    if isinstance(budget, APICallBudget):
        result["http_calls"] = budget.used
    return result


def usage_event(phase, before, after):
    usage = None
    if before is not None and after is not None:
        usage = {key: after[key] - before[key] for key in after if key != "estimated_cost"}
        if "http_calls" in usage:
            # Попытка, отклонённая бюджетом до HTTP, не является платным запросом.
            usage["requests"] = usage.pop("http_calls")
        if "estimated_cost" in after:
            old, new = before["estimated_cost"], after["estimated_cost"]
            usage["estimated_cost"] = new - old if old is not None and new is not None else None
    return {"phase": phase, "usage": usage}


def build_cost_comparison(events, decision_settings=None, *, reused_baseline=False):
    """Цены парного эксперимента с общей классификацией LLM, только текущий запуск."""
    phases = {}
    for phase in ("classification", "baseline", "verification", "enhanced"):
        entries = [event for event in events if event["phase"] == phase]
        costs, counts = [], {}
        for event in entries:
            usage = event["usage"]
            if usage is None:
                costs.append(None)
                continue
            for key, value in usage.items():
                if key != "estimated_cost":
                    counts[key] = counts.get(key, 0) + value
            if phase == "verification":
                cost = estimate_usage_costs(None, DecisionUsage(**usage),
                                            decision_settings=decision_settings, include_decision=True)["decision_usd"]
            else:
                cost = estimate_usage_costs(LLMUsage(**usage), None)["llm_usd"]
            costs.append(cost)
        phases[phase] = {"calls": len(entries), "usage": counts,
                         "cost_usd": sum(costs) if all(cost is not None for cost in costs) else None,
                         "known_subtotal_usd": sum(cost for cost in costs if cost is not None)}

    def total(names):
        values = [phases[name]["cost_usd"] for name in names]
        return sum(values) if all(value is not None for value in values) else None

    baseline = total(("classification", "baseline"))
    extra = total(("verification", "enhanced"))
    actual = total(phases)
    full = actual if not reused_baseline else None
    baseline = baseline if not reused_baseline else None
    return {"currency": "USD", "scope": "current_run_only", "reused_baseline": reused_baseline,
            "llm_only_usd": baseline, "full_usd": full, "extra_full_usd": extra,
            "full_to_llm_only_ratio": full / baseline if full is not None and baseline is not None and baseline > 0 else None,
            "audit_total_usd": actual,
            "known_subtotal_usd": sum(item["known_subtotal_usd"] for item in phases.values()),
            "phases": phases,
            "note": "Shared LLM classification and baseline, not two independent production runs. "
                    "Baseline is charged once; failed/incomplete usage or missing rates make costs unknown. "
                    "Reused/resumed baselines have unknown historical cost; audit_total covers new calls only."}


def print_cost_comparison(costs):
    def price(value):
        return f"${value:.6f}" if value is not None else "н/д"

    print("Стоимость парного сравнения (общая классификация LLM, USD):", flush=True)
    print(f"  llm-only: {price(costs['llm_only_usd'])}; full: {price(costs['full_usd'])}; "
          f"дополнительные проверки и ремонт: {price(costs['extra_full_usd'])}", flush=True)
    print(f"  Фактическая оценка расходов аудита: {price(costs['audit_total_usd'])} "
          "(общий базовый этап учтён один раз).", flush=True)
    if costs["full_to_llm_only_ratio"] is not None:
        print(f"  full / llm-only: {costs['full_to_llm_only_ratio']:.3f}×", flush=True)
    if costs["reused_baseline"]:
        print("  Цена прежней baseline неизвестна; учтены только новые запросы текущего запуска.", flush=True)
    if costs["audit_total_usd"] is None:
        print(f"  Известная часть расходов: {price(costs['known_subtotal_usd'])}; "
              "полный итог неизвестен из-за usage или тарифов.", flush=True)


def audit_llm_settings(settings, *, api_retries=2):
    """Все транспортные повторы проходят через счётчик HTTP-бюджета."""
    rate_limit_retries = settings.rate_limit_retries
    if rate_limit_retries is None:
        rate_limit_retries = settings.max_retries
    return settings.model_copy(update={"max_retries": api_retries, "rate_limit_retries": rate_limit_retries})


def temporary_api_error(error):
    return (isinstance(error, TypeSafeAPIConnectionError)
            or isinstance(error, TypeSafeAPIError) and (error.status == 408 or error.status >= 500))


def service_stop(error):
    """Авторизация и лимиты сервиса не лечатся переходом к следующему кейсу."""
    while error is not None:
        if isinstance(error, LLMRateLimitError):
            return True
        if isinstance(error, LLMError) and error.status_code in (401, 403, 429):
            return True
        if isinstance(error, TypeSafeAPIError) and error.status in (401, 403, 429):
            return True
        error = error.__cause__
    return False


class AuditDecisionModel:
    name = "Decision model"

    def __init__(self, sdk, *, request_options=None):
        self.sdk = sdk
        self.request_options = request_options or {}
        self.usage = DecisionUsage()
        self.models = set()

    @property
    def input_tokens(self):
        return self.usage.input_tokens

    @property
    def output_tokens(self):
        return self.usage.output_tokens

    def prepare_questions(self, questions):
        return questions

    async def ainvoke(self, state, questions, response_model):
        self.usage.sdk_calls += 1
        response = await self.sdk.system_one(
            state=state.model_dump(mode="json"),
            questions=self.prepare_questions(questions.model_dump(mode="json")),
            **self.request_options,
        )
        self.usage.responses += 1
        metadata = getattr(response, "usage", None)
        counts = (getattr(metadata, "input_tokens", None), getattr(metadata, "output_tokens", None))
        if all(type(count) is int and count >= 0 for count in counts):
            self.usage.usage_reports += 1
            self.usage.input_tokens += counts[0]
            self.usage.output_tokens += counts[1]
        self.models.add(response.model)
        return response_model.model_validate(response.model_dump(mode="json"))


class AuditCLM(AuditDecisionModel):
    name = "CLM"

    def prepare_questions(self, questions):
        return prepare_clm_questions(questions)


class AuditJeff(AuditDecisionModel):
    name = "Jeff"


class AuditLaya(AuditDecisionModel):
    name = "Laya"

    def __init__(self, sdk, settings):
        super().__init__(sdk, request_options=laya_request_options(settings))


async def audit_cases(cases, extractor, jev, *, max_jev_calls=10, max_attempts=2,
                      wrong_threshold=0.8, on_event=None, initial_seeds=None,
                      api_retries=2, verifier_budget=None, resume_records=None, on_usage=None):
    """Не принимает labels. Сохраняет общий исходный ответ и бюджет попыток при resume."""
    if not 0 <= api_retries <= 5:
        raise ValueError("Число сетевых повторов должно быть от 0 до 5")
    budget = verifier_budget if verifier_budget is not None else APICallBudget(max_jev_calls)
    records, stopped = [], False
    resume_records = validate_resume_records(resume_records or {}, cases)

    def emit(record):
        if on_event is not None:
            on_event(record)

    for case in cases:
        saved = resume_records.get(case.case_id)
        if saved and saved["status"] in TERMINAL_STATUSES:
            records.append(deepcopy(saved))
            emit(records[-1])
            continue
        if stopped or budget.used >= budget.maximum:
            continue  # Сохранённые завершённые пары ниже по списку всё равно попадут в отчёт.
        record = deepcopy(saved) if saved else {
            "case_id": case.case_id, "split": case.split, "before": None, "after": None,
            "messages": [message.model_dump(mode="json") for message in case.messages],
            "seed_proposal": None, "candidates": [], "jev_rounds": [], "status": "started",
        }
        records.append(record)
        stage = "baseline"
        seed = None
        for key in ("error_type", "failure_reason", "validation_failures"):
            record.pop(key, None)
        record["status"] = "started"

        async def measured(phase, client, operation, *args, **kwargs):
            before = usage_snapshot(client)
            try:
                return await operation(*args, **kwargs)
            finally:
                entry = {"case_id": case.case_id, **usage_event(phase, before, usage_snapshot(client))}
                record.setdefault("usage_events", []).append(entry)
                if on_usage is not None:
                    on_usage(entry)
                emit(record)

        def capture_seed(proposal, fields):
            nonlocal seed
            seed = proposal
            record["seed_proposal"] = proposal.model_dump(mode="json")
            record["baseline_pending_extraction"] = None
            emit(record)

        def capture_candidate(proposal, fields):
            record["candidates"].append({"proposal": proposal.model_dump(mode="json"),
                                         "checked_fields": fields.model_dump(mode="json")})
            record["pending_extraction"] = None
            emit(record)

        class LoggedExtractor:
            """Checkpoint до запроса позволяет повторить прерванный ремонт с той же базой."""
            def __init__(self, phase):
                self.phase = phase
                self.pending_key = "pending_extraction" if phase == "enhanced" else "baseline_pending_extraction"
                self.counter = phase + "_attempts"
                self.replay = deepcopy(record.get(self.pending_key))
                if self.replay is not None:
                    # Незавершённый HTTP-запрос остаётся той же смысловой попыткой.
                    record[self.counter] -= 1

            async def extract(self, messages, document_type, feedback=None, **kwargs):
                request = {"feedback": feedback, **kwargs}
                if self.replay is not None and "previous_proposal" not in kwargs:
                    request = deepcopy(self.replay)
                    if feedback:
                        request["feedback"] = feedback
                    if request.get("previous_proposal") is not None:
                        request["previous_proposal"] = ExtractionProposal.model_validate(request["previous_proposal"])
                    if request.get("previous_checked_fields") is not None:
                        fields_type = InvoiceFields if document_type == "invoice_draft" else TransferFields
                        request["previous_checked_fields"] = fields_type.model_validate(request["previous_checked_fields"])
                else:
                    self.replay = None
                record[self.counter] = record.get(self.counter, 0) + 1
                record[self.pending_key] = {
                    key: value.model_dump(mode="json") if hasattr(value, "model_dump") else value
                    for key, value in request.items()
                }
                emit(record)
                result = await measured(self.phase, extractor, extractor.extract, messages, document_type, **request)
                # При replay процессор ещё не знает базу ремонта; область защищаем здесь.
                if request.get("previous_proposal") is not None:
                    result = merge_repair(request["previous_proposal"], result, request["repair_paths"])
                record[self.phase + "_proposal"] = result.model_dump(mode="json")
                record[self.pending_key] = None
                emit(record)
                return result

        class LoggedJev:
            name = getattr(jev, "name", "Jev")
            saved_round = (deepcopy(record["jev_rounds"][-1])
                           if saved and record["jev_rounds"] and not record.get("pending_extraction") else None)

            async def ainvoke(self, state, questions, response_model):
                serialized = questions.model_dump(mode="json")
                if hasattr(jev, "prepare_questions"):
                    serialized = jev.prepare_questions(serialized)
                previous, self.saved_round = self.saved_round, None
                if previous and previous.get("questions") == serialized and "answers" in previous:
                    # Ответ уже получен до прерывания: повторно не сэмплируем решение.
                    return response_model.model_validate({"answers": previous["answers"]})
                for retry in range(api_retries + 1):
                    budget.spend()
                    entry = {"questions": serialized, "retry": retry,
                             "candidate_index": len(record["candidates"]) - 1,
                             "wrong_threshold": wrong_threshold}
                    record["jev_rounds"].append(entry)
                    emit(record)
                    try:
                        response = await measured("verification", jev, jev.ainvoke, state, questions, response_model)
                    except TypeSafeError as error:
                        entry["error_type"] = type(error).__name__
                        emit(record)
                        if not temporary_api_error(error) or retry == api_retries:
                            raise
                        if budget.used >= budget.maximum:
                            raise AuditBudgetError("Бюджет API-вызовов исчерпан") from None
                        await asyncio.sleep(min(2 ** retry, 30))
                        continue
                    entry["answers"] = response.model_dump(mode="json")["answers"]
                    emit(record)
                    return response

        try:
            initial = None
            if record.get("decision") is not None:
                decision = DocumentTypeDecision.model_validate(record["decision"])
                raw_initial = record.get("baseline_proposal") or record.get("seed_proposal")
                if raw_initial is not None:
                    initial = ExtractionProposal.model_validate(raw_initial)
            elif initial_seeds is not None and case.case_id in initial_seeds:
                decision, initial = initial_seeds[case.case_id]
            else:
                classification = await measured("classification", extractor, extractor.classify, case.messages)
                decision = DocumentTypeDecision(
                    document_type=classification.document_type, confidence=classification.confidence,
                    reason=NONE_REASON_TEXT[classification.none_reason] if classification.none_reason else None,
                )
            record["decision"] = decision.model_dump(mode="json")
            emit(record)
            if record["before"] is None:
                pending = record.get("baseline_pending_extraction")
                used = record.get("baseline_attempts", 0)
                remaining = max_attempts - used + int(pending is not None or initial is not None and used > 0)
                if remaining < 1:
                    raise PipelineRejectionError("Baseline attempts exhausted", failure_reason="invalid_extraction", validation_failures=[])
                before = await process_selected_document(
                    case.case_id, case.messages, decision, None, LoggedExtractor("baseline"), use_verifier=False,
                    max_attempts=remaining, on_candidate=capture_seed, allow_partial=False,
                    initial_proposal=None if pending is not None else initial,
                )
                record["before"] = before.model_dump(mode="json")
                emit(record)
            seed = ExtractionProposal.model_validate(record["seed_proposal"]) if decision.document_type != "none" else None
            if decision.document_type == "none":
                record.update(after=deepcopy(record["before"]), status="no_document")
            else:
                stage = record["stage"] = "enhanced"
                pending = record.get("pending_extraction")
                used = record.get("enhanced_attempts", len(record["candidates"]) or 1)
                record["enhanced_attempts"] = used
                remaining = max_attempts - used + 1
                if remaining < 1:
                    raise PipelineRejectionError("Repair attempts exhausted", failure_reason="invalid_extraction", validation_failures=[])
                latest = record.get("enhanced_proposal") or (
                    record["candidates"][-1]["proposal"] if record["candidates"] else record["seed_proposal"])
                after = await process_selected_document(
                    case.case_id, case.messages, decision, LoggedJev(), LoggedExtractor("enhanced"),
                    initial_proposal=None if pending is not None else ExtractionProposal.model_validate(latest),
                    on_candidate=capture_candidate, max_attempts=remaining, wrong_threshold=wrong_threshold,
                    allow_partial=False,
                )
                record.update(after=after.model_dump(mode="json"), status="complete")
        except AuditBudgetError:
            record["status"] = "budget_exhausted"
            stopped = True
        except TypeSafeError as error:
            # Сетевые ошибки SDK также наследуют OSError; это не сбой записи.
            record.update(status="api_error", error_type=type(error).__name__)
            stopped = service_stop(error)
        except OSError:
            raise  # Сбой сохранения не должен выглядеть как результат API-проверки.
        except Exception as error:
            # Не сохраняем текст исключений: он может содержать тело сервера/цитаты.
            record["error_type"] = type(error).__name__
            if isinstance(error, PipelineRejectionError):
                record.update(status="baseline_rejected" if stage == "baseline" else "rejected",
                              failure_reason=error.failure_reason, validation_failures=error.validation_failures)
                record["pending_extraction" if stage == "enhanced" else "baseline_pending_extraction"] = None
            elif isinstance(error, LLMResponseError):
                record.update(status="baseline_rejected", failure_reason="invalid_extraction",
                              validation_failures=[{"attempt": 1, "stage": "classification", "code": error.code}])
            elif isinstance(error, PipelineError) and stage == "enhanced" and error.__cause__ is None:
                record["status"] = "rejected"
            else:
                record["status"] = "api_error"
                stopped = service_stop(error)
        emit(record)
    return records


TERMINAL_STATUSES = {"complete", "no_document", "rejected", "baseline_rejected"}


def outcome_status(record):
    status = record.get("status", "complete" if record.get("after") is not None else "started")
    if status == "api_error" and record.get("error_type") == "PipelineRejectionError":
        return "rejected" if record.get("before") is not None else "baseline_rejected"
    return status


def validate_resume_records(records, cases):
    """Проверяет сохранённую baseline до любых новых платных вызовов."""
    result = {}
    for case in cases:
        if case.case_id not in records:
            continue
        record = deepcopy(records[case.case_id])
        if record.get("messages") != [message.model_dump(mode="json") for message in case.messages]:
            raise AuditResumeError(f"Изменился исходный диалог {case.case_id}")
        if record.get("before") is not None:
            decision = DocumentTypeDecision.model_validate(record.get("decision"))
            checked = (prepare_fields(ExtractionProposal.model_validate(record.get("seed_proposal")),
                                     case.messages, decision.document_type).model_dump(mode="json")
                       if decision.document_type != "none" else {})
            if (record["before"].get("fields") != checked
                    or record["before"].get("document_type") != decision.document_type
                    or record["before"].get("case_id") != case.case_id):
                raise AuditResumeError(f"Исходный кандидат не соответствует сохранённому результату {case.case_id}")
        record["status"] = outcome_status(record)
        if record.get("status") in {"complete", "no_document"} and (
                record.get("before") is None or record.get("after") is None
                or record["after"].get("case_id") != case.case_id):
            raise AuditResumeError(f"Нет сохранённой завершённой пары {case.case_id}")
        if record.get("candidates") and record.get("decision"):
            candidate = record["candidates"][-1]
            checked = prepare_fields(ExtractionProposal.model_validate(candidate["proposal"]), case.messages,
                                     record["decision"]["document_type"]).model_dump(mode="json")
            if checked != candidate.get("checked_fields"):
                raise AuditResumeError(f"Сохранённый кандидат не соответствует проверенным полям {case.case_id}")
        result[case.case_id] = record
    return result


def load_resume_records(path, cases):
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record["case_id"]] = record
    if not latest:
        raise AuditResumeError("В трассе нет контрольных записей")
    result = validate_resume_records(latest, cases)
    if not result:
        raise AuditResumeError("В трассе нет выбранных кейсов")
    return result


def load_initial_seeds(path, cases, *, allow_missing=False):
    """Только диалог, решение и исходный кандидат; никакие оценки/эталоны не переносятся."""
    latest = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                latest[record["case_id"]] = record
    seeds = {}
    for case in cases:
        record = latest.get(case.case_id)
        if record is not None and record.get("messages") != [message.model_dump(mode="json") for message in case.messages]:
            raise ValueError(f"Изменился исходный диалог {case.case_id}")
        if record is None or record.get("before") is None:
            if allow_missing:
                continue
            raise ValueError(f"Нет исходного результата для {case.case_id}")
        decision = DocumentTypeDecision.model_validate(record.get("decision"))
        proposal = None
        if decision.document_type != "none":
            proposal = ExtractionProposal.model_validate(record.get("seed_proposal"))
            checked = prepare_fields(proposal, case.messages, decision.document_type).model_dump(mode="json")
        else:
            checked = {}
        if record["before"].get("fields") != checked or record["before"].get("document_type") != decision.document_type:
            raise ValueError(f"Исходный кандидат не соответствует сохранённому результату {case.case_id}")
        seeds[case.case_id] = decision, proposal
    if allow_missing and not seeds:
        raise ValueError("В trace нет совместимых исходных кандидатов")
    return seeds


def build_report(records, labels, *, _include_quality=True):
    records = [{**record, "status": outcome_status(record)} for record in records]
    targets = {row["case_id"]: row for row in labels}
    selected = [targets[record["case_id"]] for record in records]
    report = {"method": "paired verification-only: shared LLM classification and identical seed",
              "primary_target": "exact field status and value; evidence metrics reported separately",
              "attempted_cases": len(records), "complete_pairs": 0, "incomplete_cases": [], "case_failures": [],
              "field_changes": {"corrected": 0, "regressed": 0, "details": []},
              "alias_aware_field_changes": {"corrected": 0, "regressed": 0, "details": []},
              "item_attribute_changes": {"corrected": 0, "regressed": 0, "details": []},
              "remaining_mismatches": [], "error_detection": calculate_detection_metrics(records, selected)}
    report["field_outcomes"] = calculate_field_outcomes(records, selected, detection=report["error_detection"])

    def correct(result, target, name, *, allow_aliases=False):
        if result is None or result["document_type"] != target["document_type"]:
            return False
        field = result.get("fields", {}).get(name, {})
        expected = target["fields"][name]
        return field_matches(target["document_type"], name, expected, field, allow_aliases=allow_aliases)

    def expected_attributes(target):
        items = target["fields"].get("items", {})
        if target["document_type"] == "invoice_draft" and items.get("status") == "fillable":
            for index, item in enumerate(items["value"]):
                for attribute in ("description", "quantity", "unit", "unit_price_rub"):
                    yield index, attribute, item[attribute]

    def correct_attribute(result, target, index, attribute, expected):
        if result is None or result["document_type"] != target["document_type"]:
            return False
        field = result.get("fields", {}).get("items", {})
        items = field.get("value")
        if field.get("status") != "fillable" or not isinstance(items, list) or index >= len(items):
            return False
        return attribute in items[index] and items[index][attribute] == expected

    for side in ("before", "after"):
        predictions = [record[side] for record in records if record[side] is not None]
        metrics = calculate_metrics(selected, predictions)
        report[side] = {"strict_metrics": metrics.model_dump(mode="json"),
                        "expected_fillable_values": metrics.fields.expected_values,
                        "correct_expected_values": metrics.fields.correct_values,
                        "total_expected_fields": metrics.fields.expected_fields,
                        "correct_field_status_values": metrics.fields.correct_status_values,
                        "expected_item_attributes": 0, "correct_item_attributes": 0,
                        "alias_aware_correct_expected_values": metrics.fields.alias_aware_correct_values}
        for record in records:
            target = targets[record["case_id"]]
            for index, attribute, expected in expected_attributes(target):
                report[side]["expected_item_attributes"] += 1
                report[side]["correct_item_attributes"] += int(correct_attribute(record[side], target, index, attribute, expected))

    for record in records:
        if record["before"] is None or record["after"] is None:
            report["incomplete_cases"].append(record["case_id"])
            report["case_failures"].append({key: record[key] for key in (
                "case_id", "status", "error_type", "failure_reason", "validation_failures",
            ) if key in record})
            continue
        report["complete_pairs"] += 1
        target = targets[record["case_id"]]
        attribute_mismatches = []
        for index, attribute, expected in expected_attributes(target):
            old, new = (correct_attribute(record[side], target, index, attribute, expected)
                        for side in ("before", "after"))
            path = f"items[{index}].{attribute}"
            if old != new:
                kind = "corrected" if new else "regressed"
                report["item_attribute_changes"][kind] += 1
                report["item_attribute_changes"]["details"].append({"case_id": record["case_id"], "field": path, "change": kind})
            if not new:
                attribute_mismatches.append(path)
        for name in target["fields"]:
            for counter, allow_aliases in (("field_changes", False), ("alias_aware_field_changes", True)):
                old, new = (correct(record[side], target, name, allow_aliases=allow_aliases)
                            for side in ("before", "after"))
                if old != new:
                    kind = "corrected" if new else "regressed"
                    report[counter][kind] += 1
                    report[counter]["details"].append({"case_id": record["case_id"], "field": name, "change": kind})
            if not correct(record["after"], target, name):
                scores = {key: answer["noul"] for round_ in record["jev_rounds"][-1:]
                          for key, answer in round_.get("answers", {}).items()
                          if key.split("::", 1)[0].split("[", 1)[0] == name}
                report["remaining_mismatches"].append({
                    "case_id": record["case_id"], "field": name,
                    "permitted_alias_only": correct(record["after"], target, name, allow_aliases=True),
                    "max_error_probability": max(scores.values(), default=None), "checks": scores,
                    **({"attribute_mismatches": attribute_mismatches} if name == "items" else {}),
                })
    delta = report["field_changes"]["corrected"] - report["field_changes"]["regressed"]
    report["conclusion"] = ("incomplete" if report["incomplete_cases"] or not records else
                            "observed_improvement_not_statistical_proof" if delta > 0 else
                            "observed_regression" if delta < 0 else "no_observed_advantage")
    if _include_quality:
        eligible = [r for r in records if r["before"] is not None
                    and r["status"] in {"complete", "no_document", "rejected"}]
        quality = build_report(eligible, labels, _include_quality=False)
        report["verification_quality"] = {
            "cases": [r["case_id"] for r in eligible],
            "api_error_cases": [r["case_id"] for r in records if r.get("status") == "api_error"],
            "budget_exhausted_cases": [r["case_id"] for r in records if r.get("status") == "budget_exhausted"],
            "baseline_rejected_cases": [r["case_id"] for r in records if r.get("status") == "baseline_rejected"],
            "rejected_cases": [r["case_id"] for r in eligible if r.get("status") == "rejected"],
            "unfinished_cases": [r["case_id"] for r in records if r["status"] == "started"],
            "note": "API/budget/baseline failures excluded; verifier/repair refusals included as lost results. Not an estimate for excluded cases.",
            "before": quality["before"], "after": quality["after"],
        }
    return report


def recover_report(trace_path: Path, labels_path: Path, output_dir: Path | None = None) -> Path:
    """Пересчитывает последние контрольные записи по текущей разметке, без API.

    Трасса содержит кандидатов и решения, но не полную статистику исходного
    запуска: usage и время не восстанавливаем из количества записей.
    """
    source = trace_path.read_bytes()
    latest = {}
    for line in source.decode("utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if (not isinstance(record, dict) or not isinstance(record.get("case_id"), str)
                or not {"before", "after", "jev_rounds", "status"} <= record.keys()):
            raise ValueError("Некорректная контрольная запись аудита")
        latest[record["case_id"]] = record
    if not latest:
        raise ValueError("В трассе нет контрольных записей")
    records = list(latest.values())
    labels = load_labels(labels_path, set(latest))
    report = build_report(records, labels)
    report.update(
        recovered_offline=True,
        source_trace_sha256=hashlib.sha256(source).hexdigest(),
        labels_sha256=hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        scoring_note="Rescored with current labels; original run metadata and unattempted cases are unknown.",
        selected_cases=list(latest), llm_usage=None, verifier_usage=None, cost_comparison=None,
        verifier_calls_recorded=sum(len(record["jev_rounds"]) for record in records),
        result_counts={
            "before": sum(record["before"] is not None for record in records),
            "after": sum(record["after"] is not None for record in records),
            "lost_results": [record["case_id"] for record in records
                             if record["before"] is not None and record["after"] is None],
            "gained_results": [record["case_id"] for record in records
                               if record["before"] is None and record["after"] is not None],
        },
    )
    directory = output_dir or trace_path.parent / (
        "recovered-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f"))
    directory.mkdir(parents=True, exist_ok=False)
    destination = directory / "report.json"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return destination


def positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Нужно положительное число")
    return parsed


def retry_count(value):
    parsed = int(value)
    if not 0 <= parsed <= 5:
        raise argparse.ArgumentTypeError("Число повторов должно быть от 0 до 5")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--live", action="store_true", help="Разрешить платные API-запросы")
    operation.add_argument("--report-trace", type=Path,
                           help="Восстановить все последние записи трассы по --labels, без API; --limit не применяется")
    parser.add_argument("--verifier", choices=("jeff", "clm", "jev", "laya", "decision"), default="jeff")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--limit", type=positive, default=5)
    parser.add_argument("--case-id", action="append", dest="case_ids",
                        help="Проверить только указанный кейс; можно повторить. Это диагностика, не общий benchmark")
    parser.add_argument("--max-verifier-calls", "--max-jev-calls", dest="max_jev_calls", type=positive, default=10)
    parser.add_argument("--max-llm-calls", type=positive, default=30)
    parser.add_argument("--api-retries", type=retry_count, default=2,
                        help="Дополнительные транспортные попытки (0..5), каждая входит в API-бюджет")
    parser.add_argument("--output-dir", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--seed-trace", type=Path,
                        help="Повторно проверить исходные кандидаты из trace.jsonl без новой baseline-генерации")
    source.add_argument("--resume-trace", type=Path,
                        help="Продолжить trace: сохранить завершённые исходы, повторить прерванные запросы")
    parser.add_argument("--allow-missing-seeds", action="store_true",
                        help="С --seed-trace: сгенерировать только отсутствующие исходные кандидаты")
    return parser


async def run_live(args):
    cases = load_cases(args.input, args.split)
    if args.case_ids:
        missing = set(args.case_ids) - {case.case_id for case in cases}
        if missing:
            raise ValueError("В выбранной выборке нет кейсов: " + ", ".join(sorted(missing)))
        cases = [case for case in cases if case.case_id in set(args.case_ids)]
    cases = cases[:args.limit]
    if not cases:
        raise ValueError("В выбранной выборке нет кейсов")
    labels = load_labels(args.labels, {case.case_id for case in cases})  # Только локальный scoring.
    if args.allow_missing_seeds and not args.seed_trace:
        raise ValueError("--allow-missing-seeds требует --seed-trace")
    initial_seeds = (load_initial_seeds(args.seed_trace, cases, allow_missing=args.allow_missing_seeds)
                     if args.seed_trace else None)
    resume_records = load_resume_records(args.resume_trace, cases) if args.resume_trace else None
    llm_settings, pipeline = LLMSettings(), PipelineSettings()
    if args.verifier == "laya":
        decision_settings = LayaSettings()
        sdk_options = laya_sdk_options(decision_settings, max_retries=0)
        verifier_type, verifier_name = AuditLaya, "Laya"
        wrong_threshold = decision_settings.wrong_threshold
    elif args.verifier == "jeff":
        decision_settings = JeffSettings()
        sdk_options = jeff_sdk_options(decision_settings, max_retries=0)
        verifier_type, verifier_name = AuditJeff, "Jeff"
        wrong_threshold = decision_settings.wrong_threshold
    elif args.verifier == "clm":
        decision_settings = CLMSettings()
        sdk_options = clm_sdk_options(decision_settings, max_retries=0)
        verifier_type, verifier_name = AuditCLM, "CLM"
        wrong_threshold = decision_settings.wrong_threshold
    else:
        decision_settings = DecisionModelSettings()
        sdk_options = decision_sdk_options(decision_settings, max_retries=0)
        verifier_type, verifier_name = AuditDecisionModel, decision_settings.model
        wrong_threshold = pipeline.wrong_threshold
    # BudgetedExtractor считает каждый HTTP-запрос, включая восстановление после 429.
    llm_settings = audit_llm_settings(llm_settings, api_retries=args.api_retries)
    configuration = {
        "verifier_provider": args.verifier, "verifier_model": decision_settings.model,
        "llm_model": llm_settings.model, "wrong_threshold": wrong_threshold,
        "max_attempts": pipeline.max_attempts,
        "llm_max_output_tokens": llm_settings.max_output_tokens,
        "llm_temperature": llm_settings.temperature,
        "pricing": {
            "llm_input_per_million_usd": llm_settings.input_price_per_million,
            "llm_output_per_million_usd": llm_settings.output_price_per_million,
            "decision_input_per_million_usd": getattr(decision_settings, "input_price_per_million", None),
            "decision_output_per_million_usd": getattr(decision_settings, "output_price_per_million", None),
        },
        "dataset_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "extraction_prompt_sha256": hashlib.sha256(json.dumps({
            kind: build_extraction_prompt(kind) for kind in ("invoice_draft", "transfer_draft")
        }, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "verification_code_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "src/verification.py").read_bytes()).hexdigest(),
        "field_rules_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "src/field_rules.py").read_bytes()).hexdigest(),
        "verifier_client_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "src/client.py").read_bytes()).hexdigest(),
        "execution_contract_sha256": hashlib.sha256(json.dumps({
            name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
            for name in ("src/llm.py", "src/pipeline.py", "src/repairs.py", "src/extraction.py",
                         "src/schemas/extraction_schemas.py", "src/schemas/document_schemas.py")
        }, sort_keys=True).encode()).hexdigest(),
        "endpoint_options_sha256": hashlib.sha256(json.dumps({
            "llm": llm_settings.base_url, "verifier": decision_settings.base_url,
            "protocol": llm_settings.api_protocol, "response_format": llm_settings.response_format,
            "request_options": laya_request_options(decision_settings) if args.verifier == "laya" else {},
        }, sort_keys=True).encode()).hexdigest(),
    }
    resume_config_verified = False
    if args.resume_trace:
        parent_config = args.resume_trace.parent / "config.json"
        if not parent_config.exists():
            parent_config = args.resume_trace.parent / "report.json"
        if parent_config.exists():
            previous = json.loads(parent_config.read_text(encoding="utf-8"))
            changed = [key for key, value in configuration.items() if key in previous and previous[key] != value]
            if changed:
                raise AuditResumeError("Настройки эксперимента изменились: " + ", ".join(changed)
                                       + "; для нового сравнения используйте --seed-trace")
            resume_config_verified = configuration.keys() <= previous.keys()
        if not resume_config_verified:
            print("Resume: в старой трассе нет полного config; диалоги и baseline проверены, совместимость всех настроек неизвестна.", flush=True)
    directory = args.output_dir or Path(".scratch") / (args.verifier + "-audit-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f"))
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(json.dumps(configuration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    budget = APICallBudget(args.max_llm_calls)
    verifier_budget = APICallBudget(args.max_jev_calls)
    usage_events = []
    saved_seed_ids = set(initial_seeds or {}) | {
        key for key, record in (resume_records or {}).items() if record.get("before") is not None
    }
    with (directory / "trace.jsonl").open("x", encoding="utf-8") as trace:
        def checkpoint(record):
            trace.write(json.dumps(record, ensure_ascii=False) + "\n")
            trace.flush()

        async with AsyncTypeSafeClient(**sdk_options) as sdk:
            jev = (verifier_type(sdk, decision_settings) if args.verifier == "laya"
                   else verifier_type(sdk))
            jev.name = verifier_name
            async with BudgetedExtractor(llm_settings, budget) as extractor:
                print(f"Парный аудит: {len(cases)} кейсов, до {args.max_jev_calls} вызовов {verifier_name} / {budget.maximum} LLM", flush=True)
                records = await audit_cases(cases, extractor, jev, max_jev_calls=args.max_jev_calls,
                                            max_attempts=pipeline.max_attempts,
                                            wrong_threshold=wrong_threshold, on_event=checkpoint,
                                            initial_seeds=initial_seeds, api_retries=args.api_retries,
                                            verifier_budget=verifier_budget, resume_records=resume_records,
                                            on_usage=usage_events.append)
                report = build_report(records, labels)
                report.update(configuration)
                report.update(selected_cases=[case.case_id for case in cases],
                              not_attempted=[case.case_id for case in cases if case.case_id not in {r["case_id"] for r in records}],
                              labels_sha256=hashlib.sha256(args.labels.read_bytes()).hexdigest(),
                              seed_trace_sha256=hashlib.sha256(args.seed_trace.read_bytes()).hexdigest() if args.seed_trace else None,
                              baseline_origin=("fresh_llm" if initial_seeds is None else
                                               "saved_trace" if len(initial_seeds) == len(cases) else
                                               "saved_and_fresh_trace"),
                              saved_seed_cases=[case.case_id for case in cases
                                                if case.case_id in saved_seed_ids],
                              fresh_seed_cases=[case.case_id for case in cases
                                                if case.case_id not in saved_seed_ids],
                              llm_http_calls=budget.used, llm_usage=asdict(extractor.usage),
                              verifier_api_calls=verifier_budget.used,
                              verifier_models=sorted(jev.models),
                              verifier_usage={"input_tokens": jev.input_tokens, "output_tokens": jev.output_tokens})
                report.update(
                    api_retries=args.api_retries, verifier_timeout_seconds=decision_settings.timeout_seconds,
                    llm_timeout_seconds=llm_settings.timeout_seconds, usage_scope="current_run_only",
                    resume_trace_sha256=hashlib.sha256(args.resume_trace.read_bytes()).hexdigest() if args.resume_trace else None,
                    resume_config_verified=resume_config_verified,
                    reused_cases=[r["case_id"] for r in records if resume_records is not None
                                  and r["case_id"] in resume_records and resume_records[r["case_id"]]["status"] in TERMINAL_STATUSES],
                    resumed_cases=[r["case_id"] for r in records if resume_records is not None
                                   and r["case_id"] in resume_records and resume_records[r["case_id"]]["status"] not in TERMINAL_STATUSES],
                )
                if args.resume_trace:
                    report["baseline_origin"] = "resumed_trace_and_missing_baselines"
                if args.verifier == "jev":
                    # Старые отчёты и читатели Jev сохраняют прежние ключи.
                    report.update(jev_api_calls=report["verifier_api_calls"], jev_models=report["verifier_models"],
                                  jev_usage=report["verifier_usage"])
                elif args.verifier == "clm":
                    report["clm_context_note"] = "Check clm-serve/vLLM context limits; encoder truncation is not reported by this API."
                elif args.verifier == "jeff":
                    report["jeff_context_note"] = "jeff-serve rejects overlong question branches with HTTP 422; default trained limit is 8192 tokens."
                elif args.verifier == "laya":
                    report["laya_context"] = laya_request_options(decision_settings)["extra_body"]
                    report["laya_context_note"] = "Token budgets are explicit; Laya may truncate state or option descriptions. Quality and thresholds require domain evaluation."
                if report["not_attempted"]:
                    report["conclusion"] = "incomplete"
                report["cost_comparison"] = {
                    **build_cost_comparison(usage_events, decision_settings,
                                             reused_baseline=bool(initial_seeds or resume_records)),
                    "attempted_cases": len(records), "selected_cases": len(cases),
                    "complete": not report["incomplete_cases"] and not report["not_attempted"],
                }
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_cost_comparison(report["cost_comparison"])
    print_detection_metrics(report["error_detection"])
    print_field_outcomes(report["field_outcomes"])
    print(f"Полных пар: {report['complete_pairs']}/{len(cases)}; исправлено полей: {report['field_changes']['corrected']}; "
          f"испорчено: {report['field_changes']['regressed']}", flush=True)
    changes = report["item_attribute_changes"]
    print(f"Атрибуты позиций отдельно (не складываются с полями): исправлено {changes['corrected']}; "
          f"испорчено {changes['regressed']}", flush=True)
    for failure in report["case_failures"]:
        print(f"  Незавершённый кейс {failure['case_id']}: {failure.get('failure_reason', failure['status'])}", flush=True)
        for item in failure.get("validation_failures", []):
            print(f"    Попытка {item['attempt']}: {item['stage']} / {item['code']}", flush=True)
    for side, title in (("before", f"До {verifier_name}"), ("after", f"После {verifier_name}")):
        values = report[side]
        print(f"{title}: верных ожидаемых значений {values['correct_expected_values']}/{values['expected_fillable_values']}; "
              f"с разрешёнными вариантами {values['alias_aware_correct_expected_values']}/{values['expected_fillable_values']}", flush=True)
        print(f"  Поля целиком (статус + значение): {values['correct_field_status_values']}/{values['total_expected_fields']}; "
              f"атрибуты позиций: {values['correct_item_attributes']}/{values['expected_item_attributes']}", flush=True)
        metrics = values["strict_metrics"]["fields"]
        precision, recall = (f"{metrics[key]:.2%}" if metrics[key] is not None else "н/д"
                             for key in ("precision", "recall"))
        print(f"  precision={precision}; recall={recall}; "
              f"лишних заполнений={metrics['false_positives']}", flush=True)
        summary = values["strict_metrics"]
        print(f"  Получено результатов: {summary['predicted_cases']}/{summary['total_cases']}; "
              f"полностью правильных документов: {summary['exact_document_correct']}/{summary['total_cases']}; "
              f"верных статусов: {metrics['correct_statuses']}/{metrics['expected_fields']}", flush=True)
        print_acceptance_metrics(AcceptanceMetrics.model_validate(summary["acceptance"]), summary["total_cases"])
    quality = report["verification_quality"]
    print(f"Качество на {len(quality['cases'])} кейсах с готовой baseline и без сбоев API/бюджета: "
          f"{quality['before']['correct_expected_values']} → {quality['after']['correct_expected_values']} "
          f"верных значений из {quality['before']['expected_fillable_values']}; "
          f"отказов проверки/ремонта: {len(quality['rejected_cases'])}", flush=True)
    print(f"Отдельно: сбоев API {len(quality['api_error_cases'])}; "
          f"некорректных baseline {len(quality['baseline_rejected_cases'])}; "
          f"остановок по бюджету {len(quality['budget_exhausted_cases'])}", flush=True)
    for item in report["remaining_mismatches"]:
        reason = "разрешённый вариант написания" if item["permitted_alias_only"] else "расхождение с разметкой"
        print(f"  {item['case_id']} {item['field']}: {reason}; максимальная оценка {verifier_name}={item['max_error_probability']}", flush=True)
        if item.get("attribute_mismatches"):
            print("    Не совпадают атрибуты: " + ", ".join(item["attribute_mismatches"]), flush=True)
    print(f"Вывод: {report['conclusion']}. Отчёт: {directory / 'report.json'}", flush=True)
    return not report["incomplete_cases"] and not report["not_attempted"]


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.report_trace:
        try:
            destination = recover_report(args.report_trace, args.labels, args.output_dir)
            recovered = json.loads(destination.read_text(encoding="utf-8"))
        except Exception as error:
            parser.exit(1, f"Восстановление отчёта остановлено: {type(error).__name__}; текст ошибки скрыт.\n")
            
        print("Отчёт восстановлен без API-запросов по текущей разметке.")
        print_detection_metrics(recovered["error_detection"])
        print_field_outcomes(recovered["field_outcomes"])
        for side, title in (("before", "До проверки"), ("after", "После проверки")):
            values = recovered[side]
            print(f"{title}: {values['correct_expected_values']}/{values['expected_fillable_values']} "
                  f"верных ожидаемых значений; результатов={recovered['result_counts'][side]}")
            print_acceptance_metrics(AcceptanceMetrics.model_validate(values["strict_metrics"]["acceptance"]),
                                     values["strict_metrics"]["total_cases"])
        print(f"Потерянные результаты: {', '.join(recovered['result_counts']['lost_results']) or 'нет'}")
        parser.exit(0, f"Отчёт: {destination}\n")
    if not args.live:
        parser.error("Для платного запуска нужен явный --live. Ключи читаются только из .env.")
    try:
        complete = asyncio.run(run_live(args))
    except AuditResumeError as error:
        parser.exit(1, f"Продолжение остановлено: {error}\n")
    except ValidationError:
        parser.exit(1, "Ошибка настроек или ответа API; значения скрыты.\n")
    except Exception as error:
        parser.exit(1, f"Аудит остановлен: {type(error).__name__}; текст ошибки скрыт.\n")
    raise SystemExit(0 if complete else 1)
