"""Оценка эксперимента на ru-invoice-extraction-benchmark только по полям из его разметки, без API.

Встроенные метрики run_experiment считают все поля счёта, включая заглушки
(placeholder_fields манифеста), поэтому для этого набора не используются.
Счета не в рублях не дают точности по amount_rub и items: для них отдельно
считается, заполнила ли система amount_rub суммой в чужой валюте и приняла ли документ.

Запуск: python -m scripts.score_ru_invoice EXPERIMENT_DIR
"""

import argparse
import json
from pathlib import Path

from scripts.audit_jev import read_latest
from scripts.build_ru_invoice_dataset import DATA, EXTERNAL_FIELDS
from src.detection_metrics import calculate_detection_metrics
from src.metrics import field_matches

MANIFEST = DATA / "ru_invoice.manifest.json"
LABELS = DATA / "labels_ru_invoice.jsonl"
SUBSETS = {
    "all": lambda meta: True,
    "test": lambda meta: meta["source_split"] == "test",
    "rub": lambda meta: meta["currency"] == "RUB",
    "non_rub": lambda meta: meta["currency"] != "RUB",
}


# Решение после ручного разбора прогона: форма бенчмарка → форма по нашему контракту
# (единственное число, src/field_rules.py), которая тоже считается верной.
# Регистр и скидка, записанная позицией, остаются ошибками.
ACCEPTED_DESCRIPTIONS = {
    "Услуги питания команды": "Услуга питания команды",
    "Этикетки самоклеящиеся для маркировки коробов": "Этикетка самоклеящаяся для маркировки коробов",
}


def effective_label(label, fields):
    """Эталон с принятой формой описания там, где ответ использует именно её."""
    expected, actual = label["fields"]["items"], (fields or {}).get("items") or {}
    answer = actual.get("value") if isinstance(actual.get("value"), list) else []
    items = [dict(item, description=ACCEPTED_DESCRIPTIONS[item["description"]])
             if item["description"] in ACCEPTED_DESCRIPTIONS and index < len(answer)
             and (answer[index] or {}).get("description") == ACCEPTED_DESCRIPTIONS[item["description"]]
             else item for index, item in enumerate(expected["value"] or [])]
    return {**label, "fields": {**label["fields"], "items": {**expected, "value": items}}}


def scored_fields(meta):
    if meta["currency"] == "RUB":
        return EXTERNAL_FIELDS
    return tuple(name for name in EXTERNAL_FIELDS if name not in ("amount_rub", "items"))


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_mode(rows, labels, meta):
    """Пул по раундам; кейс без предсказания остаётся в знаменателе."""
    result = {"planned": 0, "predicted": 0, "all_scored_correct": 0, "accepted": 0, "accepted_incorrect": 0,
              "needs_review": 0, "needs_review_incorrect": 0, "amount_rub_filled": 0,
              "amount_rub_filled_accepted": 0, "fields": {}}
    for case_id, prediction in rows:
        label = effective_label(labels[case_id], prediction["fields"] if prediction else None)
        fields = scored_fields(meta[case_id])
        result["planned"] += 1
        status = prediction["validation"]["status"] if prediction else None
        same_type = prediction is not None and prediction["document_type"] == label["document_type"]
        correct = {name: same_type and field_matches(label["document_type"], name, label["fields"][name],
                                                      prediction["fields"].get(name)) for name in fields}
        for name, ok in correct.items():
            counter = result["fields"].setdefault(name, {"correct": 0, "total": 0})
            counter["correct"] += ok
            counter["total"] += 1
        whole = all(correct.values())
        result["predicted"] += prediction is not None
        result["all_scored_correct"] += whole
        result["accepted"] += status == "passed"
        result["accepted_incorrect"] += status == "passed" and not whole
        result["needs_review"] += status == "needs_review"
        result["needs_review_incorrect"] += status == "needs_review" and not whole
        if meta[case_id]["currency"] != "RUB" and same_type:
            filled = prediction["fields"]["amount_rub"]["status"] == "fillable"
            result["amount_rub_filled"] += filled
            result["amount_rub_filled_accepted"] += filled and status == "passed"
    return result


def score_paired(records, labels, meta):
    """Первый вердикт Jev и итог ремонта по полям бенчмарка; заглушки отбрасываются."""
    result = dict.fromkeys(("tp", "fp", "fn", "tn", "corrected", "regressed", "incomplete_pairs", "skipped_cases"), 0)
    for record in records:
        # По одной записи: принятая форма описания зависит от ответа именно этой записи.
        before = (record.get("before") or {}).get("fields")
        detection = calculate_detection_metrics([record], [effective_label(labels[record["case_id"]], before)])
        result["skipped_cases"] += len(detection["skipped_cases"])
        for detail in detection["details"]:
            if detail["field"] in scored_fields(meta[detail["case_id"]]):
                result[detail["outcome"]] += 1
    for record in records:
        if record.get("before") is None or record.get("after") is None:
            result["incomplete_pairs"] += 1
            continue
        for name in scored_fields(meta[record["case_id"]]):
            correct = []
            for side in ("before", "after"):
                label = effective_label(labels[record["case_id"]], record[side]["fields"])
                correct.append(record[side]["document_type"] == label["document_type"]
                               and field_matches(label["document_type"], name, label["fields"][name],
                                                 record[side]["fields"].get(name)))
            old, new = correct
            if old != new:
                result["corrected" if new else "regressed"] += 1
    return result


def score(experiment, manifest_path=MANIFEST, labels_path=LABELS):
    experiment = Path(experiment)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    config = json.loads((experiment / "config.json").read_text(encoding="utf-8"))
    if config["input"]["sha256"] != manifest["input_sha256"]:
        raise ValueError("Эксперимент выполнен на другом входном файле, не на ru_invoice")
    meta = {row["case_id"]: row for row in manifest["cases"]}
    # Эталон — текущий выход генератора, а не копия в эксперименте: прогон эталоном не пользуется,
    # а исправление конвертера должно действовать и для уже выполненного прогона.
    planned = {row["case_id"] for row in read_rows(experiment / "labels.jsonl")}
    labels = {row["case_id"]: row for row in read_rows(Path(labels_path)) if row["case_id"] in planned}
    if set(labels) != planned:
        raise ValueError("В текущем эталоне ru_invoice нет части кейсов эксперимента")
    phases = json.loads((experiment / "report.json").read_text(encoding="utf-8"))["phases"]
    planned = list(labels)
    runs = {"llm-only": [], "full": [], "paired": []}
    for phase in phases:
        if phase["mode"] == "paired" and phase.get("report_path"):
            runs["paired"] += read_latest(Path(phase["report_path"]).parent / "trace.jsonl").values()
        elif phase["mode"] in ("llm-only", "full") and phase.get("run_directory"):
            predicted = {row["case_id"]: row for row in read_rows(Path(phase["run_directory"]) / "predictions.jsonl")}
            runs[phase["mode"]] += [(case_id, predicted.get(case_id)) for case_id in planned]
    rounds = sum(phase["mode"] == "full" for phase in phases)
    return {
        "dataset": "ru_invoice", "scored_fields": list(EXTERNAL_FIELDS), "rounds": rounds,
        "note": "Only fields labelled by the external benchmark; non-RUB amount_rub/items reported as outcomes.",
        "modes": {mode: {name: score_mode([row for row in runs[mode] if keep(meta[row[0]])], labels, meta)
                         for name, keep in SUBSETS.items()} for mode in ("llm-only", "full")},
        "paired": {name: score_paired([r for r in runs["paired"] if keep(meta[r["case_id"]])], labels, meta)
                   for name, keep in SUBSETS.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    result = score(args.experiment)
    destination = args.experiment / "ru_invoice_score.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in SUBSETS:
        cells = []
        for mode in ("llm-only", "full"):
            row = result["modes"][mode][name]
            cells.append(f"{mode}: верно {row['all_scored_correct']}/{row['planned']}, "
                         f"принято с ошибкой {row['accepted_incorrect']}")
        paired = result["paired"][name]
        print(f"{name}: " + "; ".join(cells) + f"; Jev TP/FP/FN/TN {paired['tp']}/{paired['fp']}/"
              f"{paired['fn']}/{paired['tn']}, ремонт +{paired['corrected']}/−{paired['regressed']}")
    print(f"Оценка: {destination}")


if __name__ == "__main__":
    main()
