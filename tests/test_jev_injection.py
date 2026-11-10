"""Тест Jev на внедрённых ошибках: мутации, отбор Python-проверками, метрики; без API."""

import asyncio
import json
from copy import deepcopy

import pytest

import scripts.jev_injection as injection
from src.extraction import ExtractionProposal, prepare_fields
from src.schemas.document_schemas import InvoiceFields

TEXT = "Выставь счёт ООО Кедр за 3 стула по 4000 рублей, без НДС."


def filled(value, quote):
    return {"status": "fillable", "value": value, "evidence": [{"message_id": "m1", "quote": quote}]}


@pytest.fixture
def base():
    fields = {name: {"status": "missing", "value": None, "evidence": []} for name in InvoiceFields.model_fields}
    fields["recipient_name"] = filled("ООО Кедр", "ООО Кедр")
    fields["items"] = filled([{"description": "стул", "quantity": 3, "unit": None, "unit_price_rub": 4000}],
                             "3 стула по 4000 рублей")
    fields["amount_rub"] = filled(None, "3 стула по 4000 рублей")
    fields["amount_rub"]["calculation"] = {"expression": "quantity * price", "operands": [
        {"name": "quantity", "source": {"message_id": "m1", "quote": "3"}},
        {"name": "price", "source": {"message_id": "m1", "quote": "4000"}},
    ]}
    fields["vat_mode"] = filled("no_vat", "без НДС")
    messages = [{"id": "m1", "role": "user", "text": TEXT}]
    proposal = {"fields": fields}
    prepared = prepare_fields(ExtractionProposal.model_validate(proposal),
                              injection.parse_messages(messages), "invoice_draft")
    label = {"case_id": "X001", "document_type": "invoice_draft", "fields": prepared.model_dump(mode="json")}
    return {"case_id": "X001", "document_type": "invoice_draft", "messages": messages,
            "proposal": proposal, "label": label}


def by_kind(variants):
    return {variant["kind"]: variant for variant in variants}


def test_clean_copy_matches_gold_and_every_mutation_breaks_only_its_field(base):
    variants, dropped = injection.build_variants([base])
    kinds = by_kind(variants)
    assert kinds["clean"]["errors"] == []
    for kind in ("missing_supported", "ambiguous_status", "item_attribute_missing", "noncanonical_description"):
        assert kinds[kind]["errors"] == [kinds[kind]["field"]], kind
    assert kinds["item_attribute_missing"]["field"] == "items"
    assert kinds["noncanonical_description"]["field"] == "items"


def test_mutations_change_the_candidate_as_named(base):
    kinds = by_kind(injection.build_variants([base])[0])
    missing = kinds["missing_supported"]
    assert missing["proposal"]["fields"][missing["field"]] == {
        "status": "missing", "value": None, "evidence": [], "calculation": None}
    ambiguous = kinds["ambiguous_status"]["proposal"]["fields"][kinds["ambiguous_status"]["field"]]
    assert ambiguous["status"] == "ambiguous" and ambiguous["value"] is None and ambiguous["evidence"]
    item = kinds["item_attribute_missing"]["proposal"]["fields"]["items"]["value"][0]
    assert None in (item["quantity"], item["unit_price_rub"])
    # Неканоническая форма взята из реплики пользователя: «стула» вместо «стул».
    assert kinds["noncanonical_description"]["proposal"]["fields"]["items"]["value"][0]["description"] == "стула"


@pytest.mark.parametrize("field,value", [("recipient_name", "ООО Источник"), ("recipient_inn", "7704100999")])
def test_foreign_value_takes_other_party_data_verbatim_from_the_dialogue(base, field, value):
    base["messages"].append({"id": "m2", "role": "user",
                             "text": "Для сверки наши данные продавца: ООО Источник, ИНН 7704100999."})
    variants, _ = injection.build_variants([base], kinds=("foreign_value",), fields=(field,))
    [variant] = variants
    proposed = variant["proposal"]["fields"][field]
    # Значение и цитата настоящие и прошли Python; ошибка только в принадлежности данных.
    assert proposed == {"status": "fillable", "value": value, "calculation": None,
                        "evidence": [{"message_id": "m2", "quote": value}]}
    assert variant["errors"] == [field]


def test_foreign_value_needs_other_data_in_the_dialogue(base):
    assert injection.build_variants([base], kinds=("foreign_value",)) == ([], [])


def test_mutations_caught_by_python_are_dropped_not_sent_to_jev(base):
    variants, dropped = injection.build_variants([base], kinds=("clean", "wrong_value"), fields=("amount_rub",))
    # Значение, расходящееся с формулой, ловит Python: Jev такой кандидат не получает.
    assert [variant["kind"] for variant in variants] == ["clean"]
    assert dropped == [{"case_id": "X001", "kind": "wrong_value", "field": "amount_rub", "reason": "python_caught"}]


def test_auroc_counts_ties_as_half():
    assert injection.auroc([(0.9, True), (0.1, False)]) == 1.0
    assert injection.auroc([(0.1, True), (0.9, False)]) == 0.0
    assert injection.auroc([(0.5, True), (0.5, False)]) == 0.5
    assert injection.auroc([(0.5, False)]) is None


def test_evaluation_reports_thresholds_kinds_and_clean_false_alarms():
    rows = [
        {"kind": "clean", "field": None, "fields": {"a": {"error": False, "score": 0.85}, "b": {"error": False, "score": 0.1}}},
        {"kind": "missing_supported", "field": "a", "fields": {"a": {"error": True, "score": 0.7}, "b": {"error": False, "score": 0.2}}},
        {"kind": "wrong_value", "field": "b", "fields": {"a": {"error": False, "score": 0.1}, "b": {"error": True, "score": 0.95}}},
    ]
    report = injection.evaluate(rows, threshold=0.8)
    at = {row["threshold"]: row for row in report["thresholds"]}
    assert (at[0.8]["tp"], at[0.8]["fp"], at[0.8]["fn"], at[0.8]["tn"]) == (1, 1, 1, 3)
    assert (at[0.6]["tp"], at[0.6]["fp"], at[0.6]["fn"]) == (2, 1, 0)
    assert report["kinds"]["missing_supported"] == {"variants": 1, "detected": 0, "recall": 0.0, "mean_score": 0.7}
    assert report["clean_documents"] == {"documents": 1, "flagged": 1, "false_alarm_rate": 1.0}
    # Пары «ошибка — чистое поле»: 0.7 выше трёх из четырёх, 0.95 выше всех четырёх.
    assert report["auroc"] == pytest.approx(7 / 8)


class FakeJev:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, state, questions, response_model):
        self.calls += 1
        answers = {key: {"noul": 0.9 if key.startswith("items") else 0.05} for key in questions.root}
        return response_model.model_validate({"answers": answers})


def test_scoring_links_every_check_to_its_root_field(base, tmp_path):
    variants, _ = injection.build_variants([base], kinds=("clean", "item_attribute_missing"))
    client = FakeJev()
    rows = asyncio.run(injection.score_variants(variants, client, tmp_path / "results.jsonl"))
    assert client.calls == 2
    assert {row["kind"] for row in rows} == {"clean", "item_attribute_missing"}
    assert all(row["fields"]["items"]["score"] == 0.9 for row in rows)
    assert all(row["fields"]["recipient_name"]["score"] == 0.05 for row in rows)
    saved = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["variant_id"] for row in saved] == [row["variant_id"] for row in rows]


def test_plan_without_live_sends_no_requests(base, capsys):
    variants, _ = injection.build_variants([base])
    plan = injection.plan(variants, price_per_million=0.2)
    assert plan["variants"] == len(variants) and plan["estimated_input_tokens"] > 0
    assert plan["estimated_cost_usd"] == pytest.approx(plan["estimated_input_tokens"] * 0.2 / 1e6)


class FlakyVerifier(FakeJev):
    """Отвечает с ошибкой LLM на заданных по счёту вызовах."""

    def __init__(self, failing):
        super().__init__()
        self.failing = failing

    async def ainvoke(self, state, questions, response_model):
        from src.llm import LLMResponseError
        self.calls += 1
        if self.calls in self.failing:
            raise LLMResponseError("ответ не на все вопросы")
        return response_model.model_validate({"answers": {key: {"noul": 0.1} for key in questions.root}})


def test_parallel_scoring_keeps_variant_order(base, tmp_path):
    variants, _ = injection.build_variants([base])
    rows = asyncio.run(injection.score_variants(variants, FakeJev(), tmp_path / "r.jsonl", concurrency=3))
    assert [row["variant_id"] for row in rows] == [variant["variant_id"] for variant in variants]
    saved = [json.loads(line)["variant_id"] for line in (tmp_path / "r.jsonl").read_text(encoding="utf-8").splitlines()]
    assert saved == [variant["variant_id"] for variant in variants]


def test_llm_answer_errors_are_recorded_and_two_in_a_row_stop(base, tmp_path):
    variants, _ = injection.build_variants([base])
    rows = asyncio.run(injection.score_variants(variants, FlakyVerifier({2}), tmp_path / "a.jsonl"))
    assert len(rows) == len(variants) - 1
    stopped = asyncio.run(injection.score_variants(variants, FlakyVerifier({1, 2}), tmp_path / "b.jsonl"))
    assert stopped == []


def test_verifier_choice_reuses_llm_settings_with_model_and_price_overrides(monkeypatch):
    from src.settings import LLMSettings
    monkeypatch.setattr(injection, "LLMSettings", lambda: LLMSettings(
        _env_file=None, LLM_API_KEY="KEY", LLM_MODEL="qwen/qwen3.8-27b@provider=chutes", LLM_API_PROTOCOL="openai",
        LLM_RESPONSE_FORMAT="json_schema", LLM_INPUT_PRICE_PER_MILLION=0.049, LLM_OUTPUT_PRICE_PER_MILLION=1.56))
    same = injection.llm_settings(model=None, input_price=None, output_price=None)
    assert (same.model, same.input_price_per_million) == ("qwen/qwen3.8-27b@provider=chutes", 0.049)
    other = injection.llm_settings(model="deepseek/deepseek-v4", input_price=0.3, output_price=1.2)
    assert (other.model, other.input_price_per_million, other.output_price_per_million) == ("deepseek/deepseek-v4", 0.3, 1.2)
    # Чужая модель без явных цен не наследует цены Qwen: стоимость тогда неизвестна.
    unpriced = injection.llm_settings(model="deepseek/deepseek-v4", input_price=None, output_price=None)
    assert unpriced.input_price_per_million is None and unpriced.output_price_per_million is None


def test_same_bases_restrict_candidates_to_an_earlier_report(tmp_path):
    # Сравнение проверяющих честно только на тех же кандидатах, что и прежний прогон.
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"bases": [{"case_id": "C001", "source": "run-a"}]}))
    assert injection.base_filter(report) == {("C001", "run-a")}
    assert injection.base_filter(None) is None


def test_natural_candidate_keeps_model_errors_as_ground_truth(base):
    # Естественный кандидат — ответ LLM из трассы, без мутаций; истина — расхождение с эталоном.
    record = {"case_id": "X001", "messages": base["messages"], "seed_proposal": base["proposal"],
              "before": {"document_type": "invoice_draft", "fields": {}}}
    clean = injection.natural_variant(record, base["label"], source="run-a", round_=1)
    assert (clean["kind"], clean["field"], clean["errors"]) == ("clean", None, [])
    wrong = deepcopy(base["proposal"])
    wrong["fields"]["recipient_name"]["value"] = "ООО Вектор-Плюс"
    variant = injection.natural_variant({**record, "seed_proposal": wrong}, base["label"], source="run-a", round_=2)
    assert (variant["kind"], variant["field"], variant["errors"]) == ("natural_error", "recipient_name", ["recipient_name"])
    assert variant["variant_id"] == "X001:run-a:2"
    assert injection.natural_variant({**record, "seed_proposal": None}, base["label"], source="run-a", round_=1) is None


def test_llm_verifier_gets_the_same_network_retries_as_run_experiment(monkeypatch):
    from src.settings import LLMSettings
    monkeypatch.setattr(injection, "LLMSettings", lambda: LLMSettings(
        _env_file=None, LLM_API_KEY="KEY", LLM_MODEL="qwen", LLM_API_PROTOCOL="openai"))
    settings = injection.llm_settings(model=None, input_price=None, output_price=None)
    assert (settings.max_retries, settings.rate_limit_retries) == (2, 2)


def test_resume_skips_scored_variants_and_keeps_order(base, tmp_path):
    variants, _ = injection.build_variants([base])
    path = tmp_path / "results.jsonl"
    first = asyncio.run(injection.score_variants(variants[:2], FakeJev(), path))
    path.open("a", encoding="utf-8").write(json.dumps({"variant_id": variants[2]["variant_id"], "error": "503"}) + "\n")
    done = injection.scored_rows(path)
    assert list(done) == [row["variant_id"] for row in first]
    client = FakeJev()
    rows = asyncio.run(injection.score_variants([v for v in variants if v["variant_id"] not in done], client, path))
    assert client.calls == len(variants) - 2
    merged = injection.merge_rows(variants, done, rows)
    assert [row["variant_id"] for row in merged] == [variant["variant_id"] for variant in variants]


def test_sample_keeps_every_error_and_a_fixed_set_of_clean_candidates():
    variants = [{"variant_id": f"V{i}", "kind": "natural_error" if i % 10 == 0 else "clean"} for i in range(60)]
    sample = injection.sample_clean(variants, 20)
    assert [v for v in sample if v["kind"] != "clean"] == [v for v in variants if v["kind"] != "clean"]
    assert sum(v["kind"] == "clean" for v in sample) == 20
    assert sample == injection.sample_clean(variants, 20)            # тот же набор при каждом запуске
    assert [v["variant_id"] for v in sample] == [v["variant_id"] for v in variants if v in sample]  # исходный порядок
