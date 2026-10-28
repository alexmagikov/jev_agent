"""Два раздельных эксперимента при фиксированных настройках и общем бюджете."""

import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4

from typesafe_sdk import AsyncTypeSafeClient

from scripts.audit_jev import AuditDecisionModel, audit_cases, build_report, build_cost_comparison, service_stop
from src.client import decision_sdk_options
from src.main import DEFAULT_INPUT_PATH, DEFAULT_LABELS_PATH, load_cases, load_labels, estimate_usage_costs
from src.metrics import calculate_metrics
from src.pipeline import PipelineError, process_document
from src.request_budget import APICallBudget, APICallBudgetError, BudgetedDecisionModel, BudgetedExtractor
from src.run_artifacts import RunArtifacts, _write_text_atomic, input_fingerprint, public_settings
from src.settings import DecisionModelSettings, LLMSettings, PipelineSettings

ROOT = Path(__file__).resolve().parents[1]


class ExperimentChangedError(RuntimeError):
    """Исходники или входные данные изменены после фиксации протокола."""

    def __init__(self, changed_files: list[str]):
        self.changed_files = changed_files
        self.report_path: Path | None = None
        super().__init__("После начала эксперимента изменились файлы: " + ", ".join(changed_files))


@dataclass
class Plan:
    cases: list
    labels: list
    llm: LLMSettings
    decision: DecisionModelSettings
    pipeline: PipelineSettings
    config: dict
    fingerprints: dict


def _json(path, value):
    _write_text_atomic(json.dumps(value, ensure_ascii=False, indent=2) + "\n", path)


def prepare_plan(args):
    if min(args.rounds, args.limit, args.max_llm_calls, args.max_verifier_calls) < 1 or not 0 <= args.api_retries <= 5:
        raise ValueError("Неверные лимиты эксперимента")
    cases = load_cases(args.input, args.split)
    if args.case_ids:
        if set(args.case_ids) - {case.case_id for case in cases}:
            raise ValueError("Выбранные case_id отсутствуют в указанном split")
        cases = [case for case in cases if case.case_id in set(args.case_ids)]
    cases = cases[:args.limit]
    if not cases:
        raise ValueError("В выбранной выборке нет кейсов")
    labels = load_labels(args.labels, {case.case_id for case in cases})
    llm = LLMSettings().model_copy(update={"max_retries": args.api_retries, "rate_limit_retries": args.api_retries})
    decision = DecisionModelSettings().model_copy(update={"max_retries": 0})
    pipeline = PipelineSettings()
    sources = sorted((ROOT / "src").rglob("*.py")) + [
        ROOT / "scripts/audit_jev.py", Path(__file__).resolve(), ROOT / "data/form_templates.json",
        ROOT / "pyproject.toml", ROOT / "uv.lock",
    ]
    for name in ("document-output-contract.md", "experiment-protocol.md"):
        contract = ROOT / "docs" / name
        if contract.exists():
            sources.append(contract)
    fingerprints = {path.resolve(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [*sources, args.input, args.labels]}
    config = {
        "protocol_version": 1, "python": sys.version, "rounds": args.rounds,
        "case_ids": [case.case_id for case in cases], "split": args.split,
        "input": input_fingerprint(args.input), "labels": input_fingerprint(args.labels),
        "experiments": {"modes": "independent classification and extraction in llm-only/full",
                        "paired": "shared LLM classification and candidate; strict repair audit"},
        "order": "alternate llm-only/full between rounds; paired audit follows both",
        "settings": {"llm": public_settings(llm), "decision": public_settings(decision),
                     "pipeline": {"max_attempts": pipeline.max_attempts, "wrong_threshold": pipeline.wrong_threshold}},
        "api_retries": args.api_retries,
        "budgets": {"llm": args.max_llm_calls, "verifier": args.max_verifier_calls,
                    "scope": "entire experiment, all rounds and retries"},
        "source_sha256": {str(path.relative_to(ROOT)): fingerprints[path.resolve()] for path in sources},
        "endpoints_sha256": hashlib.sha256(json.dumps([llm.base_url, decision.base_url]).encode()).hexdigest(),
        "statistics": "descriptive per-round and pooled counts; repeated cases are not independent samples",
    }
    return Plan(cases, labels, llm, decision, pipeline, config, fingerprints)


def assert_unchanged(plan):
    changed_files = []
    for path, expected in plan.fingerprints.items():
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed_files.append(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
    if changed_files:
        raise ExperimentChangedError(changed_files)


def _prefixed(row, number):
    return {**row, "case_id": f"r{number:03d}/{row['case_id']}"}


def summarize(plan, phases, budgets, *, status, stop_reason=None):
    labels = [_prefixed(label, number) for number in range(1, plan.config["rounds"] + 1) for label in plan.labels]
    modes = {mode: calculate_metrics(labels, [
        _prefixed(row, phase["round"]) for phase in phases if phase["mode"] == mode
        for row in phase.get("_predictions", [])
    ]).model_dump(mode="json") for mode in ("llm-only", "full")}
    records = []
    for phase in phases:
        if phase["mode"] != "paired":
            continue
        for record in phase.get("_records", []):
            copy = _prefixed(deepcopy(record), phase["round"])
            for side in ("before", "after"):
                if copy.get(side) is not None:
                    copy[side] = _prefixed(copy[side], phase["round"])
            records.append(copy)
    attempted = {record["case_id"] for record in records}
    not_attempted = [label["case_id"] for label in labels if label["case_id"] not in attempted]
    placeholders = [{"case_id": cid, "before": None, "after": None, "status": "not_attempted", "jev_rounds": []}
                    for cid in not_attempted]
    paired = build_report([*records, *placeholders], labels)
    paired.update(attempted_cases=len(records), not_attempted=not_attempted,
                  metrics_scope="all planned case-round pairs, including unattempted")
    costs = {}
    for mode, name in (("llm-only", "llm-only_usd"), ("full", "full_usd"), ("paired", "paired_audit_usd")):
        values = [phase.get("actual_cost_usd", 0) for phase in phases if phase["mode"] == mode]
        costs[name] = sum(values) if all(value is not None for value in values) else None
    values = list(costs.values())
    costs["actual_total_usd"] = sum(values) if all(value is not None for value in values) else None
    costs["known_subtotal_usd"] = sum(phase.get("known_cost_subtotal_usd", 0) for phase in phases)
    costs["note"] = "Actual executed requests only; the shared paired baseline is counted once. USD estimates from usage/rates."
    return {"status": status, "stop_reason": stop_reason, "calls": {name: budget.used for name, budget in budgets.items()},
            "phases": [{key: value for key, value in phase.items() if not key.startswith("_")} for phase in phases],
            "mode_comparison": modes, "paired_audit": paired, "costs": costs,
            "note": "All planned cases are in denominators. completed describes execution, not correctness of predictions."}


async def run_mode(plan, phase, directory, budgets, api_retries, *, on_start=None):
    started = perf_counter()
    mode = phase["mode"]
    verifier = BudgetedDecisionModel(plan.decision, budgets["verifier"], api_retries=api_retries) if mode == "full" else None
    extractor = BudgetedExtractor(plan.llm, budgets["llm"])
    start_calls = budgets["llm"].used
    run = RunArtifacts(directory / "predictions.jsonl", {**plan.config, "mode": mode})
    phase["run_directory"] = str(run.directory)
    results, errors = [], []
    phase["status"] = "running"

    def checkpoint():
        processed = {row.case_id for row in results} | {error["case_id"] for error in errors}
        metrics = calculate_metrics([label for label in plan.labels if label["case_id"] in processed],
                                    [row.model_dump(mode="json") for row in results])
        # Блокировка бюджетом до _send не является платным HTTP-запросом.
        usage = replace(extractor.usage, requests=budgets["llm"].used - start_calls)
        costs = estimate_usage_costs(usage, verifier.usage if verifier else None,
                                     decision_settings=plan.decision, include_decision=verifier is not None)
        run.checkpoint(results, errors, metrics, status=phase["status"], failure_kind=phase.get("stop_reason"),
                       usage={"llm": asdict(usage), "decision": asdict(verifier.usage) if verifier else None,
                              "cost_estimate": costs, "elapsed_seconds": perf_counter() - started})
        phase.update(actual_cost_usd=costs["total_usd"], elapsed_seconds=perf_counter() - started,
                     known_cost_subtotal_usd=sum(value for key, value in costs.items() if key != "total_usd" and value is not None),
                     metrics=metrics.model_dump(mode="json"), _predictions=[row.model_dump(mode="json") for row in results])

    checkpoint()
    if on_start:
        on_start()
    try:
        async with extractor:
            for case in plan.cases:
                assert_unchanged(plan)
                try:
                    result = await process_document(case.case_id, case.messages, verifier, extractor,
                        use_verifier=mode == "full", max_attempts=plan.pipeline.max_attempts,
                        wrong_threshold=plan.pipeline.wrong_threshold)
                except APICallBudgetError:
                    errors.append({"case_id": case.case_id, "error": "budget_exhausted"})
                    phase.update(status="stopped", stop_reason="budget_exhausted")
                except PipelineError as error:
                    errors.append({"case_id": case.case_id, "error": type(error).__name__})
                    if service_stop(error):
                        phase.update(status="stopped", stop_reason="service_error")
                else:
                    results.append(result)
                checkpoint()
                assert_unchanged(plan)
                if phase["status"] == "stopped":
                    break
        if phase["status"] != "stopped":
            phase["status"] = "completed"
    except BaseException as error:
        phase.update(status="interrupted", stop_reason=type(error).__name__)
        raise
    finally:
        checkpoint()


async def run_paired(plan, phase, directory, budgets, api_retries, *, on_start=None):
    directory.mkdir(parents=True, exist_ok=False)
    latest, events = {}, []
    started = perf_counter()
    phase["report_path"] = str(directory / "report.json")
    phase["status"] = "running"
    if on_start:
        on_start()
    try:
        with (directory / "trace.jsonl").open("x", encoding="utf-8") as trace:
            def checkpoint(record):
                latest[record["case_id"]] = deepcopy(record)
                trace.write(json.dumps(record, ensure_ascii=False) + "\n")
                trace.flush()
                assert_unchanged(plan)
            async with AsyncTypeSafeClient(**decision_sdk_options(plan.decision, max_retries=0)) as sdk:
                verifier = AuditDecisionModel(sdk)
                async with BudgetedExtractor(plan.llm, budgets["llm"]) as extractor:
                    await audit_cases(plan.cases, extractor, verifier, max_attempts=plan.pipeline.max_attempts,
                        wrong_threshold=plan.pipeline.wrong_threshold, api_retries=api_retries,
                        verifier_budget=budgets["verifier"], on_event=checkpoint, on_usage=events.append)
        if any(record.get("status") == "budget_exhausted" for record in latest.values()):
            phase.update(status="stopped", stop_reason="budget_exhausted")
        elif any(record.get("status") == "api_error" for record in latest.values()) or len(latest) < len(plan.cases):
            phase.update(status="stopped", stop_reason="api_error")
        else:
            phase["status"] = "completed"
    except BaseException as error:
        phase.update(status="interrupted", stop_reason=type(error).__name__)
        raise
    finally:
        phase["_records"] = list(latest.values())
        report = build_report(phase["_records"], plan.labels)
        report["cost_comparison"] = build_cost_comparison(events, plan.decision)
        phase.update(actual_cost_usd=report["cost_comparison"]["audit_total_usd"],
                     known_cost_subtotal_usd=report["cost_comparison"]["known_subtotal_usd"],
                     elapsed_seconds=perf_counter() - started)
        _json(directory / "report.json", report)


async def run_experiment(args):
    plan = prepare_plan(args)
    if not args.live:
        preview = {key: value for key, value in plan.config.items() if key != "source_sha256"}
        preview["snapshot_source_files"] = len(plan.config["source_sha256"])
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        print("Это план без запросов. Для запуска используйте --live. Лимиты вызовов не являются денежным потолком.")
        return None
    assert_unchanged(plan)
    root = args.output_dir or Path("experiments") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8])
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    _json(root / "config.json", plan.config)
    for name in plan.config["source_sha256"]:
        target = root / "snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    protocol = ROOT / "docs/experiment-protocol.md"
    if protocol.exists():
        (root / "protocol.md").write_bytes(protocol.read_bytes())
    for filename, rows in (("cases.jsonl", [case.model_dump(mode="json") for case in plan.cases]), ("labels.jsonl", plan.labels)):
        (root / filename).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    phases = [{"round": number, "mode": mode, "status": "not_started"}
              for number in range(1, args.rounds + 1)
              for mode in (("llm-only", "full", "paired") if number % 2 else ("full", "llm-only", "paired"))]
    budgets = {"llm": APICallBudget(args.max_llm_calls), "verifier": APICallBudget(args.max_verifier_calls)}
    status, stop_reason = "running", None
    changed_files = []
    destination = root / "report.json"

    def save():
        report = summarize(plan, phases, budgets, status=status, stop_reason=stop_reason)
        if changed_files:
            report["changed_files"] = changed_files
        _json(destination, report)

    save()
    try:
        for phase in phases:
            assert_unchanged(plan)
            if any(budget.used >= budget.maximum for budget in budgets.values()):
                status, stop_reason = "stopped", "budget_exhausted"
                break
            print(f"Раунд {phase['round']}/{args.rounds}: {phase['mode']}", flush=True)
            phase["status"] = "running"
            save()
            directory = root / f"round-{phase['round']:03d}" / phase["mode"]
            if phase["mode"] == "paired":
                await run_paired(plan, phase, directory, budgets, args.api_retries, on_start=save)
            else:
                await run_mode(plan, phase, directory, budgets, args.api_retries, on_start=save)
            save()
            if phase["status"] == "stopped":
                status, stop_reason = "stopped", phase["stop_reason"]
                break
        else:
            assert_unchanged(plan)
            status = "completed"
    except BaseException as error:
        status, stop_reason = "interrupted", type(error).__name__
        if isinstance(error, ExperimentChangedError):
            changed_files = error.changed_files
            error.report_path = destination
        raise
    finally:
        save()
    final = json.loads(destination.read_text(encoding="utf-8"))
    print(f"Эксперимент: {status}. Вызовы: {final['calls']}. Стоимость USD: {final['costs']['actual_total_usd']}")
    print(f"Отчёт: {destination}")
    return destination


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Разрешить платные запросы; без флага показывается только план")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--split", choices=("all", "dev", "test"), default="dev")
    parser.add_argument("--case-id", dest="case_ids", action="append")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument("--max-llm-calls", type=int, default=60)
    parser.add_argument("--max-verifier-calls", type=int, default=30)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv=None):
    try:
        path = asyncio.run(run_experiment(build_parser().parse_args(argv)))
    except ExperimentChangedError as error:
        message = str(error)
        if error.report_path is not None:
            message += f"\nЧастичные результаты сохранены. Отчёт: {error.report_path}"
        raise SystemExit(message) from None
    except Exception as error:
        raise SystemExit(f"Эксперимент остановлен: {type(error).__name__}; подробности ответа API не выводятся.") from None
    if path and json.loads(path.read_text(encoding="utf-8"))["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
