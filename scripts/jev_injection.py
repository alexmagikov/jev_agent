"""Тест Jev на внедрённых ошибках.

Берёт правильные кандидаты из трасс парного аудита на dev-данных (evaluation50
исключён), вносит в копии по одной контролируемой ошибке и спрашивает Jev теми же
вопросами, что в пайплайне. Мутации, которые ловит Python, Jev не получает.
Истина для каждого поля — сравнение с эталоном, а не тип мутации.

Без --live печатается план и оценка стоимости; запросов нет.
"""

import argparse
import asyncio
import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from src.cases.document_type import DocumentMessage
from src.client import DecisionModelClient, DecisionModelError, decision_sdk_options
from src.llm import AsyncLLMExtractor, LLMError
from src.llm_verifier import LLMDecisionClient
from src.extraction import ExtractionError, ExtractionProposal, prepare_fields
from src.metrics import exact_document, field_matches
from src.schemas.document_schemas import field_root
from src.settings import DecisionModelSettings, LLMSettings
from src.verification import (
    _COMMON_RULES, _RULES, VerificationResponse, VerificationState, build_verification_questions, parse_check_key,
)

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("clean", "missing_supported", "ambiguous_status", "wrong_value", "foreign_value",
         "item_attribute_missing", "noncanonical_description")
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
# Правдоподобные чужие значения: их нет в диалогах, формат при этом корректен.
DECOYS = {"recipient_name": "ООО Вектор-Плюс", "destination_bank": "Банк Восток",
          "subject": "аренда помещения", "recipient_relation": "коллега",
          "source_account_kind": "кредитная карта"}
# Данные другой стороны того же формата: в диалоге есть, но к полю не относятся.
FOREIGN_PATTERNS = {
    "recipient_name": re.compile(r"\b(?:ООО|АО|ПАО|ЗАО|ИП)\s+(?:«[^»]+»|\"[^\"]+\"|[А-ЯЁ][\w-]+(?:\s+[А-ЯЁ][\w-]+){0,2})"),
    "recipient_inn": re.compile(r"(?<!\d)(?:\d{12}|\d{10})(?!\d)"),
    "recipient_account": re.compile(r"(?<!\d)\d{20}(?!\d)"),
    "recipient_bik": re.compile(r"(?<!\d)\d{9}(?!\d)"),
}
MAX_CONSECUTIVE_ERRORS = 2


def parse_messages(messages: list[dict]) -> list[DocumentMessage]:
    return [DocumentMessage.model_validate(message) for message in messages]


def base_filter(report: Path | None) -> set[tuple[str, str]] | None:
    """Пары (кейс, прогон) из прежнего отчёта: те же кандидаты для сравнения проверяющих."""
    if report is None:
        return None
    return {(base["case_id"], base["source"]) for base in json.loads(report.read_text(encoding="utf-8"))["bases"]}


def load_bases(experiments: Path, only: set[tuple[str, str]] | None = None) -> list[dict]:
    """Последний правильный кандидат каждого кейса из трасс dev-прогонов (или ровно пары only)."""
    bases = {}
    for trace in sorted(experiments.glob("*/round-*/paired/trace.jsonl")):
        run = trace.parents[2]
        if only is not None and run.name not in {source for _, source in only}:
            continue
        if "evaluation50" in json.loads((run / "config.json").read_text(encoding="utf-8"))["input"]["path"]:
            continue
        labels = {row["case_id"]: row for row in map(json.loads, (run / "labels.jsonl").read_text(
            encoding="utf-8").splitlines()) if row}
        records = {}
        for line in trace.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["case_id"]] = record  # в трассе последняя запись — итог кейса
        for case_id, record in records.items():
            before, label = record.get("before"), labels.get(case_id)
            if only is not None and (case_id, run.name) not in only:
                continue
            if (record.get("seed_proposal") and before and label and before["document_type"] != "none"
                    and exact_document(label, before)):
                bases[case_id] = {"case_id": case_id, "document_type": before["document_type"],
                                  "messages": record["messages"], "proposal": record["seed_proposal"],
                                  "label": label, "source": run.name}
    return [bases[case_id] for case_id in sorted(bases)]


def natural_variant(record: dict, label: dict, *, source: str, round_: int) -> dict | None:
    """Кандидат LLM из трассы парного аудита как есть; ошибки — расхождения с эталоном."""
    before, proposal = record.get("before"), record.get("seed_proposal")
    if not proposal or not before or before["document_type"] != label["document_type"] or label["document_type"] == "none":
        return None
    messages = parse_messages(record["messages"])
    try:
        prepared = prepare_fields(ExtractionProposal.model_validate(proposal), messages, label["document_type"])
    except (ExtractionError, ValidationError):
        return None
    errors = _errors({"label": label}, prepared.model_dump(mode="json"))
    return {"variant_id": f"{record['case_id']}:{source}:{round_}", "case_id": record["case_id"],
            "kind": "natural_error" if errors else "clean", "field": errors[0] if errors else None,
            "errors": errors, "document_type": label["document_type"], "messages": record["messages"],
            "proposal": proposal}


def load_natural(experiments: Path, dataset: str = "cases_challenge") -> list[dict]:
    """Все исходные кандидаты парного аудита на выборке dataset; одинаковые кандидаты кейса — один раз."""
    variants, seen = [], set()
    for trace in sorted(experiments.glob("*/round-*/paired/trace.jsonl")):
        run = trace.parents[2]
        if dataset not in json.loads((run / "config.json").read_text(encoding="utf-8"))["input"]["path"]:
            continue
        labels = {row["case_id"]: row for row in map(json.loads, (run / "labels.jsonl").read_text(
            encoding="utf-8").splitlines()) if row}
        records = {}
        for line in trace.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["case_id"]] = record
        round_ = int(trace.parents[1].name.split("-")[1])
        for case_id, record in records.items():
            variant = natural_variant(record, labels[case_id], source=run.name, round_=round_)
            if variant is None:
                continue
            key = (case_id, json.dumps(record["messages"], ensure_ascii=False), json.dumps(variant["proposal"], sort_keys=True))
            if key not in seen:
                seen.add(key)
                variants.append(variant)
    return variants


def _pick(case_id: str, kind: str, options: list):
    """Детерминированный выбор: один и тот же набор при каждом запуске."""
    if not options:
        return None
    digest = hashlib.sha256(f"{case_id}:{kind}".encode()).digest()
    return options[int.from_bytes(digest[:4], "big") % len(options)]


def _inflected_description(description: str, messages: list[dict]) -> str | None:
    """Форма первого слова описания из реплики пользователя («стул» -> «стула»)."""
    first, _, rest = description.partition(" ")
    stem = first.lower() if len(first) <= 4 else first.lower()[:-2]
    words = (word.strip(".,:;!?«»\"()").lower() for message in messages if message["role"] == "user"
             for word in message["text"].split())
    form = next((word for word in words if word.startswith(stem) and word != first.lower()), None)
    return None if form is None else " ".join(filter(None, (form, rest)))


def _mutate(base: dict, kind: str, allowed: tuple[str, ...] | None) -> tuple[str, dict] | None:
    proposal = deepcopy(base["proposal"])
    fields = proposal["fields"]
    fillable = [name for name, field in fields.items() if field["status"] == "fillable"
                and (allowed is None or name in allowed)]
    if kind in ("missing_supported", "ambiguous_status"):
        name = _pick(base["case_id"], kind, fillable)
        if name is None:
            return None
        status = "missing" if kind == "missing_supported" else "ambiguous"
        evidence = [] if status == "missing" else fields[name]["evidence"]
        fields[name] = {"status": status, "value": None, "evidence": evidence, "calculation": None}
        return name, proposal
    if kind == "wrong_value":
        scalar = [name for name in fillable if name != "items"]
        name = _pick(base["case_id"], kind, scalar)
        if name is None:
            return None
        field = fields[name]
        value = field["value"] if field["value"] is not None else base["label"]["fields"][name]["value"]
        if name in DECOYS:
            field["value"] = DECOYS[name]
        elif isinstance(value, bool) or value is None:
            return None
        elif isinstance(value, (int, float)):
            field["value"] = value * 2
        elif name == "vat_mode":
            field["value"] = "rate" if value == "no_vat" else "no_vat"
        elif isinstance(value, str) and value[-1:].isdigit():
            field["value"] = value[:-1] + str((int(value[-1]) + 1) % 10)
        else:
            field["value"] = value + " (уточнить)"
        return name, proposal
    if kind == "foreign_value":
        options = []
        for name, pattern in FOREIGN_PATTERNS.items():
            if name not in fields or (allowed is not None and name not in allowed):
                continue
            gold = str(base["label"]["fields"][name].get("value") or "").casefold()
            for message in base["messages"]:
                if message["role"] != "user":
                    continue
                for match in pattern.finditer(message["text"]):
                    value = match.group(0)
                    if not gold or (value.casefold() not in gold and gold not in value.casefold()):
                        options.append((name, value, message["id"]))
        choice = _pick(base["case_id"], kind, sorted(set(options)))
        if choice is None:
            return None
        name, value, message_id = choice
        fields[name] = {"status": "fillable", "value": value, "calculation": None,
                        "evidence": [{"message_id": message_id, "quote": value}]}
        return name, proposal
    items = fields.get("items")
    if items is None or items["status"] != "fillable" or (allowed is not None and "items" not in allowed):
        return None
    if kind == "item_attribute_missing":
        options = [(index, attribute) for index, item in enumerate(items["value"])
                   for attribute in ("quantity", "unit_price_rub") if item.get(attribute) is not None]
        choice = _pick(base["case_id"], kind, options)
        if choice is None:
            return None
        index, attribute = choice
        items["value"][index][attribute] = None
        items["value"][index][f"{'quantity' if attribute == 'quantity' else 'unit_price'}_calculation"] = None
        return "items", proposal
    if kind == "noncanonical_description":
        options = [(index, form) for index, item in enumerate(items["value"]) if item.get("description")
                   for form in [_inflected_description(item["description"], base["messages"])] if form]
        choice = _pick(base["case_id"], kind, options)
        if choice is None:
            return None
        items["value"][choice[0]]["description"] = choice[1]
        return "items", proposal
    return None


def _errors(base: dict, prepared: dict) -> list[str]:
    label = base["label"]
    return [name for name, expected in label["fields"].items() if not field_matches(
        label["document_type"], name, expected, prepared.get(name), allow_aliases=True)]


def build_variants(bases: list[dict], *, kinds: tuple[str, ...] = KINDS,
                   fields: tuple[str, ...] | None = None) -> tuple[list[dict], list[dict]]:
    """Чистая копия и мутации каждого кейса; мутации, пойманные Python, отбрасываются."""
    variants, dropped = [], []
    for base in bases:
        messages = parse_messages(base["messages"])
        for kind in kinds:
            if kind == "clean":
                name, proposal = None, deepcopy(base["proposal"])
            else:
                mutated = _mutate(base, kind, fields)
                if mutated is None:
                    continue
                name, proposal = mutated
            try:
                prepared = prepare_fields(ExtractionProposal.model_validate(proposal), messages, base["document_type"])
            except (ExtractionError, ValidationError):
                if kind == "clean":
                    dropped.append({"case_id": base["case_id"], "kind": kind, "field": None, "reason": "clean_rejected"})
                    break
                dropped.append({"case_id": base["case_id"], "kind": kind, "field": name, "reason": "python_caught"})
                continue
            errors = _errors(base, prepared.model_dump(mode="json"))
            if (kind == "clean") != (not errors):
                reason = "clean_differs_from_gold" if kind == "clean" else "no_effect"
                dropped.append({"case_id": base["case_id"], "kind": kind, "field": name, "reason": reason})
                if kind == "clean":
                    break
                continue
            variants.append({"variant_id": f"{base['case_id']}:{kind}", "case_id": base["case_id"], "kind": kind,
                             "field": name, "errors": errors, "document_type": base["document_type"],
                             "messages": base["messages"], "proposal": proposal})
    return variants, dropped


def _request(variant: dict):
    messages = parse_messages(variant["messages"])
    proposal = ExtractionProposal.model_validate(variant["proposal"])
    prepared = prepare_fields(proposal, messages, variant["document_type"])
    questions = build_verification_questions(prepared, proposal=proposal, document_type=variant["document_type"],
                                             messages=messages)
    rules = _RULES if variant["document_type"] == "invoice_draft" else _COMMON_RULES
    return VerificationState(messages=messages, document_type=variant["document_type"], rules=rules), questions


async def _score(variant: dict, client) -> dict:
    state, questions = _request(variant)
    try:
        response = await client.ainvoke(state=state, questions=questions, response_model=VerificationResponse)
    except (DecisionModelError, LLMError) as error:
        return {"variant_id": variant["variant_id"], "error": str(error)}
    checks: dict[str, dict[str, float]] = {}
    for key, answer in response.answers.items():
        checks.setdefault(field_root(parse_check_key(key)[0]), {})[key] = answer.noul
    row = {key: variant[key] for key in ("variant_id", "case_id", "kind", "field", "document_type")}
    row["fields"] = {name: {"error": name in variant["errors"], "score": max(scores.values()),
                            "checks": scores} for name, scores in checks.items()}
    return row


async def score_variants(variants: list[dict], client, path: Path, *, max_calls: int | None = None,
                         concurrency: int = 1) -> list[dict]:
    """По одному запросу на вариант; concurrency запросов одновременно, порядок вариантов сохраняется.

    Два сбоя подряд (по порядку вариантов) останавливают прогон, как в run_experiment.
    """
    rows, consecutive, pending = [], 0, variants[:max_calls]
    with path.open("a", encoding="utf-8") as stream:
        for start in range(0, len(pending), concurrency):
            batch = await asyncio.gather(*(_score(variant, client) for variant in pending[start:start + concurrency]))
            for row in batch:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                if "error" in row:
                    consecutive += 1
                    if consecutive >= MAX_CONSECUTIVE_ERRORS:
                        return rows
                    continue
                consecutive = 0
                rows.append(row)
            stream.flush()
    return rows


def sample_clean(variants: list[dict], count: int) -> list[dict]:
    """Все варианты с ошибками и count чистых, выбранных по хешу ID: одинаково при каждом запуске."""
    clean = sorted((variant for variant in variants if variant["kind"] == "clean"),
                   key=lambda variant: hashlib.sha256(f"20261028:{variant['variant_id']}".encode()).hexdigest())
    chosen = {variant["variant_id"] for variant in clean[:count]}
    return [variant for variant in variants if variant["kind"] != "clean" or variant["variant_id"] in chosen]


def scored_rows(path: Path) -> dict[str, dict]:
    """Успешные ответы прежнего запуска: при продолжении их не запрашивают снова."""
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {row["variant_id"]: row for row in rows if "fields" in row}


def merge_rows(variants: list[dict], done: dict[str, dict], new: list[dict]) -> list[dict]:
    by_id = done | {row["variant_id"]: row for row in new}
    return [by_id[variant["variant_id"]] for variant in variants if variant["variant_id"] in by_id]


def auroc(pairs: list[tuple[float, bool]]) -> float | None:
    """Вероятность, что поле с ошибкой получит оценку выше чистого; ничья — половина."""
    positives = [score for score, error in pairs if error]
    negatives = [score for score, error in pairs if not error]
    if not positives or not negatives:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def evaluate(rows: list[dict], *, threshold: float = 0.8) -> dict:
    pairs = [(field["score"], field["error"]) for row in rows for field in row["fields"].values()]
    table = []
    for cut in THRESHOLDS:
        tp = sum(score >= cut and error for score, error in pairs)
        fp = sum(score >= cut and not error for score, error in pairs)
        fn = sum(score < cut and error for score, error in pairs)
        tn = sum(score < cut and not error for score, error in pairs)
        table.append({"threshold": cut, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                      "precision": tp / (tp + fp) if tp + fp else None,
                      "recall": tp / (tp + fn) if tp + fn else None,
                      "false_positive_rate": fp / (fp + tn) if fp + tn else None})
    kinds = {}
    for row in rows:
        if row["kind"] == "clean":
            continue
        score = row["fields"].get(row["field"], {}).get("score", 0.0)
        entry = kinds.setdefault(row["kind"], {"variants": 0, "detected": 0, "scores": []})
        entry["variants"] += 1
        entry["detected"] += score >= threshold
        entry["scores"].append(score)
    kinds = {kind: {"variants": entry["variants"], "detected": entry["detected"],
                    "recall": entry["detected"] / entry["variants"],
                    "mean_score": round(sum(entry["scores"]) / len(entry["scores"]), 4)}
             for kind, entry in sorted(kinds.items())}
    clean = [row for row in rows if row["kind"] == "clean"]
    flagged = sum(any(field["score"] >= threshold for field in row["fields"].values()) for row in clean)
    return {"threshold": threshold, "fields": len(pairs), "error_fields": sum(error for _, error in pairs),
            "auroc": auroc(pairs), "thresholds": table, "kinds": kinds,
            "clean_documents": {"documents": len(clean), "flagged": flagged,
                                "false_alarm_rate": flagged / len(clean) if clean else None}}


def plan(variants: list[dict], *, price_per_million: float | None) -> dict:
    """Грубая оценка: ~3 символа JSON на токен плюс ~280 служебных токенов на запрос."""
    tokens = 0
    for variant in variants:
        state, questions = _request(variant)
        tokens += len(state.model_dump_json()) // 3 + len(questions.model_dump_json()) // 3 + 280
    counts = {kind: sum(variant["kind"] == kind for variant in variants) for kind in KINDS}
    return {"variants": len(variants), "by_kind": counts, "estimated_input_tokens": tokens,
            "estimated_cost_usd": None if price_per_million is None else tokens * price_per_million / 1e6}


def _print_report(metrics: dict) -> None:
    print(f"Поля: {metrics['fields']}, с ошибкой: {metrics['error_fields']}, AUROC: {metrics['auroc']}")
    for row in metrics["thresholds"]:
        print(f"  порог {row['threshold']}: TP {row['tp']}, FP {row['fp']}, FN {row['fn']}, TN {row['tn']}, "
              f"полнота {row['recall']}, точность {row['precision']}")
    for kind, entry in metrics["kinds"].items():
        print(f"  {kind}: найдено {entry['detected']}/{entry['variants']}, средняя оценка {entry['mean_score']}")
    clean = metrics["clean_documents"]
    print(f"  Чистые документы с ложной пометкой: {clean['flagged']}/{clean['documents']}")


def llm_settings(*, model: str | None, input_price: float | None, output_price: float | None,
                 retries: int = 2) -> LLMSettings:
    """Настройки извлекающей LLM из .env; другая модель не наследует её цены.

    Сетевые повторы — как у LLM в run_experiment (--api-retries 2): 503 провайдера не останавливает прогон сразу.
    """
    settings = LLMSettings()
    update = {"max_retries": retries, "rate_limit_retries": retries}
    if model is not None and model != settings.model:
        update |= {"model": model, "input_price_per_million": None, "output_price_per_million": None}
    if input_price is not None:
        update["input_price_per_million"] = input_price
    if output_price is not None:
        update["output_price_per_million"] = output_price
    return settings.model_copy(update=update)


async def run(args) -> Path:
    decision = DecisionModelSettings()
    if args.natural:
        bases, dropped = [], []
        variants = load_natural(args.experiments)
        if args.sample_clean is not None:
            variants = sample_clean(variants, args.sample_clean)
    else:
        bases = load_bases(args.experiments, base_filter(args.same_bases))
        variants, dropped = build_variants(bases)
    llm = llm_settings(model=args.llm_model, input_price=args.llm_input_price, output_price=args.llm_output_price) \
        if args.verifier == "llm" else None
    price = decision.input_price_per_million if llm is None else llm.input_price_per_million
    name = decision.model if llm is None else llm.model
    estimate = plan(variants, price_per_million=price)
    note = "" if llm is None else " (только вход; выход LLM с рассуждениями не оценивается)"
    print(f"Проверяющий: {name}. Кейсов: {len(bases)}; вариантов: {estimate['variants']} {estimate['by_kind']}; "
          f"отброшено: {len(dropped)}; оценка ~{estimate['estimated_input_tokens']} входных токенов, "
          f"${estimate['estimated_cost_usd']}{note}")
    if not args.live:
        print("Без --live запросы не отправляются.")
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "natural" if args.natural else "jev-injection"
    output = args.resume or args.output_dir or args.experiments / f"{prefix}-{stamp}-{args.verifier}"
    output.mkdir(parents=True, exist_ok=args.resume is not None)
    done = scored_rows(output / "results.jsonl") if args.resume else {}
    if not done:
        with (output / "variants.jsonl").open("w", encoding="utf-8") as stream:
            stream.writelines(json.dumps(variant, ensure_ascii=False) + "\n" for variant in variants)
    pending = [variant for variant in variants if variant["variant_id"] not in done]
    print(f"Уже оценено: {len(done)}; осталось: {len(pending)}")
    started = datetime.now(timezone.utc)
    if llm is None:
        options = decision_sdk_options(decision, max_retries=2)
        client = DecisionModelClient(api_key=options.pop("api_key"), sdk_options=options, name=decision.model)
        rows = await score_variants(pending, client, output / "results.jsonl", max_calls=args.max_calls,
                                    concurrency=args.concurrency)
        usage = {"calls": client.usage.sdk_calls, "input_tokens": client.usage.input_tokens,
                 "output_tokens": client.usage.output_tokens, "errors": client.usage.error_responses}
        cost = client.usage.input_tokens * price / 1e6 if price is not None else None
    else:
        async with AsyncLLMExtractor(llm) as extractor:
            client = LLMDecisionClient(extractor, name=llm.model)
            rows = await score_variants(pending, client, output / "results.jsonl", max_calls=args.max_calls,
                                        concurrency=args.concurrency)
        usage = {"calls": extractor.usage.requests, "input_tokens": extractor.usage.input_tokens,
                 "output_tokens": extractor.usage.output_tokens, "errors": extractor.usage.error_responses}
        cost = extractor.usage.estimated_cost
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    rows = merge_rows(variants, done, rows)
    metrics = evaluate(rows, threshold=args.threshold)
    report = {"created": stamp, "verifier": args.verifier, "model": name, "threshold": args.threshold,
              "concurrency": args.concurrency, "elapsed_seconds": round(elapsed, 1),
              "bases": [{"case_id": base["case_id"], "source": base["source"]} for base in bases],
              "dropped": dropped, "plan": estimate, "scored": len(rows),
              "failed": len(variants) - len(rows), "resumed_from_scored": len(done), "usage": usage, "cost_usd": cost,
              "metrics": metrics}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_report(metrics)
    print(f"Проверяющий {name}: стоимость ${cost}, время {elapsed / 60:.1f} мин; отчёт: {output / 'report.json'}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=ROOT / "experiments")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--live", action="store_true", help="Отправить платные запросы")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--verifier", choices=("jev", "llm"), default="jev",
                        help="jev — модель решений из DECISION_*; llm — LLM из LLM_* с теми же вопросами")
    parser.add_argument("--llm-model", help="Другая модель для --verifier llm (по умолчанию LLM_MODEL)")
    parser.add_argument("--llm-input-price", type=float, help="USD за 1M входных токенов для --llm-model")
    parser.add_argument("--llm-output-price", type=float, help="USD за 1M выходных токенов для --llm-model")
    parser.add_argument("--concurrency", type=int, default=1, help="Одновременных запросов")
    parser.add_argument("--same-bases", type=Path, help="report.json прежнего прогона: взять те же кандидаты")
    parser.add_argument("--resume", type=Path, help="Каталог прерванного запуска: оценить только оставшиеся варианты")
    parser.add_argument("--sample-clean", type=int, help="С --natural: все ошибочные и столько чистых кандидатов")
    parser.add_argument("--natural", action="store_true",
                        help="Вместо мутаций — исходные кандидаты LLM из прогонов на challenge (естественные ошибки)")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
