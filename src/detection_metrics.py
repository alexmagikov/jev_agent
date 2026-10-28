"""Оценка обнаружения ошибок по первой проверке исходного кандидата в трассе аудита."""

from collections.abc import Mapping, Sequence
from math import isfinite

from src.metrics import JsonRow, field_matches


def calculate_detection_metrics(records: Sequence[JsonRow], labels: Sequence[JsonRow]) -> dict:
    """ Cравнивает исходный кандидат с эталоном и смотрит первый ответ Jev. 
    Считает правильные обнаружения, ложные замечания, пропущенные ошибки и правильные принятия — TP, FP, FN, TN.

    Истина — статус и значение эталона с разрешёнными вариантами. Для оценки
    источников или конкретных причин ошибки такой разметки недостаточно.
    Старые трассы без связи ответа с кандидатом или порога не угадываются.
    """

    targets = {label["case_id"]: label for label in labels}
    report = {"version": 1, "unit": "root_field", "ground_truth": "status/value with permitted aliases",
              "total_cases": len(records), "evaluated_cases": 0, "evaluated_fields": 0,
              "total_fields": sum(len(targets[r["case_id"]]["fields"]) for r in records),
              "tp": 0, "fp": 0, "fn": 0, "tn": 0, "details": [], "skipped_cases": []}
    
    for record in records:
        case_id = record["case_id"]
        target, before = targets[case_id], record.get("before")
        rounds = record.get("jev_rounds", [])
        first = next((r for r in rounds if r.get("candidate_index") == 0 and "answers" in r), None)
        candidates = record.get("candidates", [])
        reason = None
        
        if before is None:
            reason = "no_baseline"
        elif before["document_type"] != target["document_type"]:
            reason = "wrong_document_type"
        elif before["document_type"] == "none":
            reason = "no_document"
        elif first is None:
            reason = "unlinked_initial_verdict" if any("answers" in r for r in rounds) else "no_initial_verdict"
        elif not candidates or candidates[0].get("checked_fields") != before.get("fields"):
            reason = "candidate_mismatch"
        else:
            threshold = first.get("wrong_threshold")
            if not _number(threshold) or not 0 < threshold < 1:
                reason = "unknown_threshold"
    
        if reason is not None:
            report["skipped_cases"].append({"case_id": case_id, "reason": reason})
            continue

        scores = _round_scores(first, target["fields"])
        if not scores:
            report["skipped_cases"].append({"case_id": case_id, "reason": "invalid_initial_verdict"})
            continue

        report["evaluated_cases"] += 1
        for name, checks in scores.items():
            error = not field_matches(target["document_type"], name, target["fields"][name],
                                      before["fields"].get(name), allow_aliases=True)
            flagged_checks = [key for key, value in checks.items() if value >= threshold]
            flagged = bool(flagged_checks)
            outcome = "tp" if error and flagged else "fn" if error else "fp" if flagged else "tn"
            report[outcome] += 1
            report["evaluated_fields"] += 1
            report["details"].append({"case_id": case_id, "field": name, "error_present": error,
                                      "flagged": flagged, "outcome": outcome,
                                      "max_error_probability": max(checks.values()), "threshold": threshold,
                                      "flagged_checks": flagged_checks})
    flagged = report["tp"] + report["fp"]
    erroneous = report["tp"] + report["fn"]
    report.update(
        precision=report["tp"] / flagged if flagged else None,
        recall=report["tp"] / erroneous if erroneous else None,
        field_coverage=report["evaluated_fields"] / report["total_fields"] if report["total_fields"] else None,
        unassessed_fields=report["total_fields"] - report["evaluated_fields"],
    )
    return report


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _root(path: str) -> str:
    return path.split("::", 1)[0].split("[", 1)[0].split(".", 1)[0]


def _round_scores(round_: Mapping, fields: Mapping) -> dict:
    questions, answers = round_.get("questions"), round_.get("answers")
    if (not isinstance(questions, Mapping) or not isinstance(answers, Mapping)
            or not questions or set(questions) != set(answers)):
        return {}
    scores = {}
    for key, answer in answers.items():
        root = _root(key)
        value = answer.get("noul") if isinstance(answer, Mapping) else None
        if "::" not in key or root not in fields or not _number(value) or not 0 <= value <= 1:
            return {}
        scores.setdefault(root, {})[key] = value
    return scores


def calculate_field_outcomes(records: Sequence[JsonRow], labels: Sequence[JsonRow], *, detection: Mapping) -> dict:
    """Связывает первый вердикт с итогом того же поля; содержимое и маршрут могут пересекаться."""
    targets = {label["case_id"]: label for label in labels}
    sources = {record["case_id"]: record for record in records}
    names = ("corrected", "detected_and_review", "missed", "regressed_by_repair", "false_review",
             "detected_and_rejected", "false_rejection", "withheld_fields", "other_review", "unknown_final")
    report = {"version": 1, "unit": "root_field", "counters_overlap": True,
              **dict.fromkeys(names, 0), "linked_fields": len(detection["details"]),
              "unassessed_fields": detection["unassessed_fields"], "details": []}
    for initial in detection["details"]:
        case_id, name = initial["case_id"], initial["field"]
        state = _final_field_state(sources[case_id], targets[case_id], name)
        disposition = state["final_disposition"]
        semantic_review = disposition == "needs_review" and "semantic" in state["review_stages"]
        conditions = {
            "corrected": initial["error_present"] and state["final_correct"] is True,
            "regressed_by_repair": not initial["error_present"] and state["final_correct"] is False,
            "detected_and_review": semantic_review and state["final_correct"] is False,
            "false_review": semantic_review and state["final_correct"] is True,
            "missed": disposition == "passed" and state["final_correct"] is False,
            "detected_and_rejected": disposition == "rejected" and state["last_candidate_correct"] is False,
            "false_rejection": disposition == "rejected" and state["last_candidate_correct"] is True,
            "withheld_fields": disposition == "withheld",
            "other_review": disposition == "needs_review" and not semantic_review,
            "unknown_final": disposition == "unknown",
        }
        outcomes = [key for key, applies in conditions.items() if applies]
        for key in outcomes:
            report[key] += 1
        report["details"].append({"case_id": case_id, "field": name,
                                  "initial_error": initial["error_present"], "initial_flagged": initial["flagged"],
                                  "initial_outcome": initial["outcome"], **state, "outcomes": outcomes})
    return report


def _final_field_state(record: Mapping, target: Mapping, name: str) -> dict:
    state = {"final_correct": None, "last_candidate_correct": None, "final_disposition": "unknown",
             "document_disposition": record.get("status", "unknown"), "review_origin": None, "review_stages": []}
    after = record.get("after")
    if after is not None:
        actual = after.get("fields", {}).get(name)
        state["final_correct"] = (after["document_type"] == target["document_type"]
                                  and field_matches(target["document_type"], name, target["fields"][name],
                                                    actual, allow_aliases=True))
        raw_validation = after.get("validation")
        validation = raw_validation if isinstance(raw_validation, Mapping) else {}
        document_status = validation.get("status", "unknown")
        state["document_disposition"] = document_status
        issues = validation.get("issues", [])
        stages = sorted({issue.get("stage", "unknown") for issue in issues if _root(issue.get("path", "")) == name})
        marked = actual is not None and actual.get("validation_status") == "needs_review"
        if marked or stages:
            state.update(final_disposition="needs_review", review_stages=stages,
                         review_origin=stages[0] if len(stages) == 1 else "mixed" if stages else "unknown")
        elif document_status in {"passed", "needs_review"} and not any(not issue.get("path") for issue in issues):
            state["final_disposition"] = "passed"
        return state

    # Строгий отказ не является ни опубликованным полем, ни направлением человеку.
    if record.get("status") != "rejected" or record.get("failure_reason") != "semantic_rejection":
        return state
    candidates, rounds = record.get("candidates", []), record.get("jev_rounds", [])
    if not candidates or not rounds or rounds[-1].get("candidate_index") != len(candidates) - 1:
        return state
    last = rounds[-1]
    threshold = last.get("wrong_threshold")
    scores = _round_scores(last, target["fields"])
    if name not in scores or not _number(threshold) or not 0 < threshold < 1:
        return state
    checked = candidates[-1].get("checked_fields", {})
    if name not in checked:
        return state
    state["last_candidate_correct"] = field_matches(target["document_type"], name, target["fields"][name],
                                                     checked[name], allow_aliases=True)
    if any(value >= threshold for value in scores[name].values()):
        state.update(final_disposition="rejected", review_origin="semantic", review_stages=["semantic"])
    else:
        state["final_disposition"] = "withheld"
    return state


def print_field_outcomes(metrics: Mapping) -> None:
    print("Итог по полям, связанным с первой проверкой (счётчики могут пересекаться):")
    print(f"  Исправлено: {metrics['corrected']}; обнаружено и отправлено на проверку: {metrics['detected_and_review']}; "
          f"ошибочных полей без замечания в итоге: {metrics['missed']}")
    print(f"  Испорчено после ремонта: {metrics['regressed_by_repair']}; "
          f"ошибочно отправлено на проверку: {metrics['false_review']}")
    print(f"  Строго отклонено ошибочных полей: {metrics['detected_and_rejected']}; "
          f"ошибочно отклонено: {metrics['false_rejection']}; соседних полей не выдано: {metrics['withheld_fields']}")
    print(f"  Проверка по другим причинам: {metrics['other_review']}; итог неизвестен: {metrics['unknown_final']}; "
          f"полей без исходной оценки: {metrics['unassessed_fields']}")


def print_detection_metrics(metrics: Mapping) -> None:
    """Вывод одинаков для нового аудита и восстановления сохранённой трассы."""
    def percent(value):
        return f"{value:.1%}" if value is not None else "н/д"

    print("Обнаружение ошибок в первой проверке исходного кандидата (корневые поля):")
    print(f"  TP={metrics['tp']}; FP={metrics['fp']}; FN={metrics['fn']}; TN={metrics['tn']}")
    print(f"  Точность замечаний: {percent(metrics['precision'])}; "
          f"полнота обнаружения: {percent(metrics['recall'])}")
    print(f"  Оценено полей: {metrics['evaluated_fields']}/{metrics['total_fields']} "
          f"({percent(metrics['field_coverage'])}); без оценки: {metrics['unassessed_fields']}; "
          f"пропущено кейсов: {len(metrics['skipped_cases'])}")
