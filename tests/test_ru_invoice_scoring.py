"""Оценка прогона на внешнем бенчмарке только по его полям; синтетический эксперимент, без API."""

from copy import deepcopy
import json

import pytest

from scripts import build_ru_invoice_dataset as builder
from scripts import score_ru_invoice as scoring


RUB, USD = "invoice-001", "invoice-042"


def dataset():
    labels = {json.loads(line)["case_id"]: json.loads(line)
              for line in (builder.DATA / "labels_ru_invoice.jsonl").read_text(encoding="utf-8").splitlines()}
    manifest = json.loads((builder.DATA / "ru_invoice.manifest.json").read_text(encoding="utf-8"))
    return labels, manifest


def prediction(label, status="passed", **changes):
    fields = deepcopy(label["fields"])
    for name, value in changes.items():
        fields[name] = {"status": "fillable", "value": value, "evidence": ["m1"], "derivation": None}
    return {"case_id": label["case_id"], "document_type": "invoice_draft", "fields": fields,
            "validation": {"status": status, "issues": []}}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def experiment(tmp_path):
    labels, manifest = dataset()
    root = tmp_path / "experiment"
    write_rows(root / "labels.jsonl", [labels[RUB], labels[USD]])
    (root / "config.json").write_text(json.dumps({"input": {"sha256": manifest["input_sha256"]}}))
    rub, usd = labels[RUB], labels[USD]
    # Заглушка recipient_account расходится с «эталоном», но в оценку не входит.
    write_rows(root / "llm" / "predictions.jsonl", [
        prediction(rub, recipient_account="40702810000000020002"),
        prediction(usd, amount_rub=4867.8),
    ])
    write_rows(root / "full" / "predictions.jsonl", [
        prediction(rub, recipient_inn="0000000000"),
        prediction(usd, status="needs_review"),
    ])
    before = prediction(rub, amount_rub=1.0)
    keys = {"amount_rub::superseded": .9, "recipient_account::unsupported": .95, "recipient_name::unsupported": .1}
    record = {"case_id": RUB, "before": before, "after": prediction(rub), "status": "complete",
              "candidates": [{"checked_fields": before["fields"]}],
              "jev_rounds": [{"candidate_index": 0, "wrong_threshold": .8, "questions": dict.fromkeys(keys, {}),
                              "answers": {key: {"noul": value} for key, value in keys.items()}}]}
    write_rows(root / "paired" / "trace.jsonl", [record])
    (root / "report.json").write_text(json.dumps({"phases": [
        {"round": 1, "mode": "llm-only", "run_directory": str(root / "llm")},
        {"round": 1, "mode": "full", "run_directory": str(root / "full")},
        {"round": 1, "mode": "paired", "report_path": str(root / "paired" / "report.json")},
    ]}))
    return root


def test_only_benchmark_fields_are_scored(experiment):
    result = scoring.score(experiment)
    llm, full = (result["modes"][mode]["rub"] for mode in ("llm-only", "full"))
    assert (llm["planned"], llm["all_scored_correct"], llm["accepted_incorrect"]) == (1, 1, 0)
    assert (full["all_scored_correct"], full["accepted_incorrect"]) == (0, 1)
    assert full["fields"]["recipient_inn"] == {"correct": 0, "total": 1}
    assert "recipient_account" not in llm["fields"]


def test_foreign_currency_amounts_are_reported_as_their_own_outcome(experiment):
    result = scoring.score(experiment)
    llm, full = (result["modes"][mode]["non_rub"] for mode in ("llm-only", "full"))
    assert "amount_rub" not in llm["fields"] and "items" not in llm["fields"]
    assert (llm["amount_rub_filled"], llm["amount_rub_filled_accepted"]) == (1, 1)
    assert (full["amount_rub_filled"], full["needs_review"]) == (0, 1)


def test_jev_detection_and_repair_count_benchmark_fields_only(experiment):
    paired = scoring.score(experiment)["paired"]["all"]
    assert (paired["tp"], paired["fp"], paired["fn"], paired["tn"]) == (1, 0, 0, 1)
    assert paired["corrected"] == 1 and paired["regressed"] == 0


def test_scoring_uses_the_current_dataset_labels_not_the_copy_saved_with_the_run(experiment):
    labels, _ = dataset()
    stale = deepcopy(labels[RUB])
    stale["fields"]["recipient_name"]["value"] = "Старый эталон"
    write_rows(experiment / "labels.jsonl", [stale, labels[USD]])
    assert scoring.score(experiment)["modes"]["llm-only"]["rub"]["all_scored_correct"] == 1


def test_reviewed_singular_description_counts_but_letter_case_does_not():
    labels, manifest = dataset()
    meta = {row["case_id"]: row for row in manifest["cases"]}
    label = labels["invoice-050"]
    index = next(i for i, item in enumerate(label["fields"]["items"]["value"])
                 if item["description"] == "Услуги питания команды")

    def with_description(text):
        result = prediction(label)
        result["fields"]["items"]["value"][index]["description"] = text
        return result

    rows = [("invoice-050", with_description("Услуга питания команды")),
            ("invoice-050", with_description("услуги питания команды"))]
    result = scoring.score_mode(rows, labels, meta)
    assert (result["planned"], result["all_scored_correct"], result["accepted_incorrect"]) == (2, 1, 1)


def test_experiment_on_another_dataset_is_rejected(experiment):
    (experiment / "config.json").write_text(json.dumps({"input": {"sha256": "0" * 64}}))
    with pytest.raises(ValueError, match="другом"):
        scoring.score(experiment)
