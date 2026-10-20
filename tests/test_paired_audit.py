import asyncio
import json

import pytest

from scripts.audit_jev import APICallBudget, AuditBudgetError, AuditJev, BudgetedExtractor, audit_cases, build_report, build_parser
from src.extraction import ExtractionProposal
from src.main import InputCase
from src.schemas.document_schemas import TransferFields
from src.schemas.extraction_schemas import DocumentClassification


def case(case_id="C1"):
    return InputCase.model_validate({"case_id": case_id, "split": "dev", "messages": [
        {"id": "m1", "role": "user", "text": "Переведи себе 100 рублей"},
        {"id": "m2", "role": "user", "text": "Исправление: 200 рублей"},
    ]})


def proposal(value):
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in TransferFields.model_fields}
    fields["amount_rub"] = {"status": "fillable", "value": value,
                            "evidence": [{"message_id": "m1" if value == 100 else "m2", "quote": f"{value} рублей"}]}
    return ExtractionProposal.model_validate({"fields": fields})


class Extractor:
    def __init__(self):
        self.classifications = 0
        self.extractions = []

    async def classify(self, messages, feedback=None):
        self.classifications += 1
        return DocumentClassification(document_type="transfer_draft", confidence=0.9, none_reason=None)

    async def extract(self, messages, document_type, feedback=None, **kwargs):
        self.extractions.append(kwargs)
        return proposal(100 if len(self.extractions) == 1 else 200)


class Jev:
    def __init__(self):
        self.states = []

    async def ainvoke(self, state, questions, response_model):
        self.states.append(state)
        assert "GOLD_ONLY_MARKER" not in repr((state, questions))
        return response_model.model_validate({"answers": {
            key: {"noul": 0.99 if key == "amount_rub::superseded" and questions.root[key].instructions["field"]["value"] == 100 else 0.01}
            for key in questions.root
        }})


def label(value=200):
    fields = {name: {"status": "missing", "value": None, "evidence": [], "derivation": None}
              for name in TransferFields.model_fields}
    fields["amount_rub"].update(status="fillable", value=value, evidence=["m2"])
    return {"case_id": "C1", "document_type": "transfer_draft", "fields": fields,
            "scoring_only": "GOLD_ONLY_MARKER"}


def test_audit_reuses_exact_seed_and_scores_both_directions_without_gold_leakage():
    extractor, jev = Extractor(), Jev()
    records = asyncio.run(audit_cases([case()], extractor, jev))
    assert extractor.classifications == 1
    assert len(extractor.extractions) == 2  # исходная генерация + ремонт, не новая baseline
    record = records[0]
    assert record["before"]["fields"]["amount_rub"]["value"] == 100
    assert record["after"]["fields"]["amount_rub"]["value"] == 200
    assert record["candidates"][0]["proposal"] == record["seed_proposal"]
    assert len(record["jev_rounds"]) == 2
    assert extractor.extractions[1]["repair_fields"] == ["amount_rub"]
    report = build_report(records, [label()])
    assert report["field_changes"]["corrected"] == 1
    assert report["field_changes"]["regressed"] == 0
    assert report["before"]["correct_expected_values"] == 0
    assert report["after"]["correct_expected_values"] == 1
    assert "GOLD_ONLY_MARKER" not in json.dumps(records)


def test_report_exposes_regressions_instead_of_proving_a_desired_result():
    records = asyncio.run(audit_cases([case()], Extractor(), Jev()))
    report = build_report(records, [label(100)])
    assert report["field_changes"]["corrected"] == 0
    assert report["field_changes"]["regressed"] == 1
    assert report["after"]["correct_expected_values"] < report["before"]["correct_expected_values"]


def test_budget_exhaustion_is_reported_and_does_not_become_a_successful_fallback():
    extractor, jev = Extractor(), Jev()
    records = asyncio.run(audit_cases([case(), case("C2")], extractor, jev, max_jev_calls=1))
    assert len(jev.states) == 1
    assert len(records) == 1
    assert records[0]["status"] == "budget_exhausted"
    assert records[0]["after"] is None
    report = build_report(records, [label()])
    assert report["complete_pairs"] == 0
    assert report["incomplete_cases"] == ["C1"]


def test_api_errors_are_sanitized_and_stop_further_paid_calls():
    class UnavailableJev:
        async def ainvoke(self, *args):
            raise RuntimeError("DO_NOT_PRINT_SECRET")

    records = asyncio.run(audit_cases([case(), case("C2")], Extractor(), UnavailableJev()))
    assert len(records) == 1 and records[0]["status"] == "api_error"
    assert records[0]["after"] is None
    assert "DO_NOT_PRINT_SECRET" not in json.dumps(records)


@pytest.mark.parametrize(("kind", "code", "stage"), [
    ("llm", "output_token_limit", "llm_response"),
    ("python", "unsupported_numeric_value", "python_validation"),
])
def test_rejected_repair_records_safe_validation_reason_without_an_accepted_fallback(kind, code, stage):
    from src.extraction import ExtractionError
    from src.llm import LLMResponseError

    class InvalidRepair(Extractor):
        async def extract(self, *args, **kwargs):
            if kwargs.get("repair_fields"):
                error_type = LLMResponseError if kind == "llm" else ExtractionError
                raise error_type("DO_NOT_PRINT_SECRET", code=code)
            return proposal(100)

    records = asyncio.run(audit_cases([case()], InvalidRepair(), Jev()))
    record = records[0]
    assert record["status"] == "rejected"
    assert record["after"] is None
    assert record["failure_reason"] == "invalid_extraction"
    assert record["validation_failures"] == [{"attempt": 2, "stage": stage, "code": code}]
    assert "DO_NOT_PRINT_SECRET" not in json.dumps(records)
    report = build_report(records, [label()])
    assert report["conclusion"] == "incomplete"
    assert report["case_failures"][0]["validation_failures"] == record["validation_failures"]


def test_repeated_semantic_rejection_is_distinct_from_an_invalid_repair_response():
    class StillWrong(Extractor):
        async def extract(self, *args, **kwargs):
            return proposal(100)

    record = asyncio.run(audit_cases([case()], StillWrong(), Jev()))[0]
    assert record["status"] == "rejected" and record["after"] is None
    assert record["failure_reason"] == "semantic_rejection"
    assert record["validation_failures"] == []


def test_budget_and_cli_require_explicit_paid_opt_in():
    budget = APICallBudget(1)
    budget.spend()
    with pytest.raises(AuditBudgetError):
        budget.spend()
    args = build_parser().parse_args([])
    assert not args.live
    assert args.max_jev_calls == 10
    assert args.max_llm_calls == 30
    assert args.limit == 5


@pytest.mark.parametrize(("configured", "expected"), [(None, 2), (0, 0), (2, 2)])
def test_audit_uses_explicit_counted_429_retries_without_network_retries(configured, expected):
    from scripts.audit_jev import audit_llm_settings
    from src.settings import LLMSettings

    settings = LLMSettings(_env_file=None, LLM_API_KEY="fake", LLM_MODEL="small",
                           LLM_MAX_RETRIES=2, LLM_RATE_LIMIT_RETRIES=configured)
    bounded = audit_llm_settings(settings)
    assert bounded.max_retries == 0
    assert bounded.rate_limit_retries == expected


@pytest.mark.parametrize("maximum", [1, 2])
def test_429_recovery_cannot_exceed_audit_http_budget(monkeypatch, maximum):
    import httpx2
    from scripts.audit_jev import audit_llm_settings
    from src.settings import LLMSettings

    calls = []
    async def sleep(delay):
        pass
    monkeypatch.setattr("src.llm.asyncio.sleep", sleep)
    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx2.Response(429, headers={"Retry-After": "0"})
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "document_type": "none", "confidence": 0.9, "none_reason": "no_request",
        })}}]})
    async def run():
        settings = audit_llm_settings(LLMSettings(_env_file=None, LLM_API_KEY="fake", LLM_MODEL="small",
                                                 LLM_RATE_LIMIT_RETRIES=2))
        budget = APICallBudget(maximum)
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with BudgetedExtractor(settings, budget) as extractor:
                extractor._client = client
                if maximum == 1:
                    with pytest.raises(AuditBudgetError):
                        await extractor.classify(case().messages)
                else:
                    assert (await extractor.classify(case().messages)).document_type == "none"
                assert budget.used == maximum
    asyncio.run(run())
    assert len(calls) == maximum


def test_llm_budget_is_checked_before_the_underlying_http_request(monkeypatch):
    from src.llm import AsyncLLMExtractor
    from src.settings import LLMSettings

    requests = []

    async def send(self, payload):
        requests.append(payload)
        return b"{}"

    monkeypatch.setattr(AsyncLLMExtractor, "_send", send)

    async def run():
        extractor = BudgetedExtractor(LLMSettings(_env_file=None, LLM_API_KEY="fake", LLM_MODEL="small"), APICallBudget(1))
        assert await extractor._send({}) == b"{}"
        with pytest.raises(AuditBudgetError):
            await extractor._send({})

    asyncio.run(run())
    assert len(requests) == 1


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_token_limit_failure_has_a_safe_code_from_the_real_response_parser(protocol):
    from src.llm import AsyncLLMExtractor, LLMResponseError
    from src.settings import LLMSettings

    settings = LLMSettings(_env_file=None, LLM_API_KEY="fake", LLM_MODEL="small",
                           LLM_API_PROTOCOL=protocol, LLM_RESPONSE_FORMAT="text")
    extractor = AsyncLLMExtractor(settings)
    raw = ({"content": [{"type": "text", "text": "PRIVATE_MODEL_TEXT"}], "stop_reason": "max_tokens"}
           if protocol == "anthropic" else
           {"choices": [{"finish_reason": "length", "message": {"content": "PRIVATE_MODEL_TEXT"}}]})
    with pytest.raises(LLMResponseError) as raised:
        extractor._parse_response(json.dumps(raw).encode())
    assert raised.value.code == "output_token_limit"
    assert "PRIVATE_MODEL_TEXT" not in str(raised.value)


def test_unsupported_direct_total_has_a_safe_code_from_real_python_validation():
    from src.extraction import ExtractionError, prepare_fields

    unsupported = proposal(200)
    unsupported.fields["amount_rub"].value = 300
    with pytest.raises(ExtractionError) as raised:
        prepare_fields(unsupported, case().messages, "transfer_draft")
    assert raised.value.code == "unsupported_numeric_value"


@pytest.mark.parametrize("kind", ["llm", "python"])
def test_unknown_diagnostic_codes_cannot_contain_arbitrary_error_text(kind):
    from src.extraction import ExtractionError
    from src.llm import LLMResponseError

    error = (LLMResponseError if kind == "llm" else ExtractionError)(
        "PRIVATE_MESSAGE", code="PRIVATE_CUSTOM_CODE",
    )
    assert error.code == ("invalid_response" if kind == "llm" else "invalid_proposal")


def test_unknown_target_case_is_rejected_before_any_api_setup(tmp_path, monkeypatch):
    import scripts.audit_jev as module

    path = tmp_path / "cases.jsonl"
    path.write_text(case().model_dump_json())
    monkeypatch.setattr(module, "LLMSettings", lambda: pytest.fail("API setup must not run"))
    args = build_parser().parse_args(["--live", "--input", str(path), "--case-id", "UNKNOWN"])
    with pytest.raises(ValueError, match="UNKNOWN"):
        asyncio.run(module.run_live(args))


def test_jev_audit_collects_real_response_usage_without_a_second_sdk_request():
    from types import SimpleNamespace
    from src.verification import VerificationResponse

    calls = []

    class SDK:
        async def system_one(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(usage=SimpleNamespace(input_tokens=12, output_tokens=3), model="test-model",
                                   model_dump=lambda **kwargs: {"answers": {"amount_rub::unsupported": {"noul": 0.1}}})

    class Data:
        def model_dump(self, **kwargs):
            return {}

    client = AuditJev(SDK())
    response = asyncio.run(client.ainvoke(Data(), Data(), VerificationResponse))
    assert response.answers["amount_rub::unsupported"].noul == 0.1
    assert len(calls) == 1
    assert client.input_tokens == 12 and client.output_tokens == 3
    assert client.models == {"test-model"}


def test_empty_experiment_cannot_claim_advantage():
    assert build_report([], [])["conclusion"] == "incomplete"


def test_audit_counts_rejected_unsupported_value_even_when_fillable_accuracy_does_not_change():
    records = asyncio.run(audit_cases([case()], Extractor(), Jev()))
    record = records[0]
    record["after"] = json.loads(json.dumps(record["before"]))
    record["after"]["fields"]["amount_rub"].update(status="ambiguous", value=None)
    target = label()
    target["fields"]["amount_rub"].update(status="ambiguous", value=None)
    report = build_report(records, [target])
    assert report["before"]["expected_fillable_values"] == report["after"]["expected_fillable_values"] == 0
    assert report["before"]["correct_field_status_values"] == 4
    assert report["after"]["correct_field_status_values"] == 5
    assert report["before"]["total_expected_fields"] == 5
    assert report["field_changes"]["corrected"] == 1


def test_partial_item_repairs_and_regressions_are_counted_separately_from_whole_fields():
    def item(description, unit):
        return {"description": description, "quantity": 2, "unit": unit, "unit_price_rub": 700}

    def result(items):
        return {"document_type": "invoice_draft", "case_id": "local", "fields": {
            "items": {"status": "fillable", "value": items, "evidence": [], "derivation": None}}}

    target = result([item("ремонт", None)])
    record = {"case_id": "local", "before": result([item("ремонты", "ремонтов")]),
              "after": result([item("ремонты", None)]), "jev_rounds": []}
    report = build_report([record], [target])
    assert report["field_changes"]["corrected"] == 0
    assert report["item_attribute_changes"]["corrected"] == 1
    assert report["item_attribute_changes"]["regressed"] == 0
    assert report["item_attribute_changes"]["details"][0]["field"] == "items[0].unit"
    assert report["before"]["correct_item_attributes"] == 2
    assert report["after"]["correct_item_attributes"] == 3
    assert report["after"]["expected_item_attributes"] == 4
    record["before"], record["after"] = record["after"], record["before"]
    inverse = build_report([record], [target])
    assert inverse["item_attribute_changes"]["regressed"] == 1
    assert inverse["item_attribute_changes"]["corrected"] == 0


def test_missing_or_wrongly_routed_item_does_not_count_null_attributes_as_correct():
    target = {"case_id": "local", "document_type": "invoice_draft", "fields": {"items": {
        "status": "fillable", "value": [{"description": "ремонт", "quantity": None,
                                         "unit": None, "unit_price_rub": None}]}}}
    missing = {**target, "fields": {"items": {"status": "missing", "value": None}}}
    record = {"case_id": "local", "before": missing, "after": None, "jev_rounds": []}
    report = build_report([record], [target])
    assert report["before"]["correct_item_attributes"] == report["after"]["correct_item_attributes"] == 0
    assert report["after"]["expected_item_attributes"] == 4
    assert report["item_attribute_changes"]["corrected"] == 0
    assert report["conclusion"] == "incomplete"


def test_audit_distinguishes_permitted_aliases_from_missed_semantic_errors():
    records = asyncio.run(audit_cases([case()], Extractor(), Jev()))
    # Один и тот же плохой исходный ответ принят проверяющим: низкие оценки не скрывают ошибку.
    record = records[0]
    record["after"] = json.loads(json.dumps(record["before"]))
    record["before"]["fields"]["destination_bank"].update(status="fillable", value="Сбербанк")
    record["after"]["fields"]["destination_bank"].update(status="fillable", value="Сбербанк")
    record["jev_rounds"] = [{"answers": {"amount_rub::superseded": {"noul": 0.19}}}]
    target = label()
    target["fields"]["destination_bank"].update(status="fillable", value="Сбер")
    report = build_report(records, [target])
    assert report["before"]["correct_expected_values"] == 0
    assert report["before"]["alias_aware_correct_expected_values"] == 1
    assert report["conclusion"] == "no_observed_advantage"
    assert report["alias_aware_field_changes"]["corrected"] == 0
    remaining = {item["field"]: item for item in report["remaining_mismatches"]}
    assert remaining["destination_bank"]["permitted_alias_only"] is True
    assert remaining["amount_rub"]["permitted_alias_only"] is False
    assert remaining["amount_rub"]["max_error_probability"] == 0.19


def test_saved_seed_recheck_skips_classification_and_baseline_generation():
    from src.pipeline import DocumentTypeDecision

    class RepairOnly(Extractor):
        async def classify(self, *args, **kwargs):
            pytest.fail("Saved-seed audit must not reclassify")

        async def extract(self, *args, **kwargs):
            assert kwargs["repair_fields"] == ["amount_rub"]
            self.extractions.append(kwargs)
            return proposal(200)

    extractor = RepairOnly()
    seeds = {"C1": (DocumentTypeDecision(document_type="transfer_draft", confidence=0.9), proposal(100))}
    record = asyncio.run(audit_cases([case()], extractor, Jev(), initial_seeds=seeds))[0]
    assert len(extractor.extractions) == 1
    assert record["before"]["fields"]["amount_rub"]["value"] == 100
    assert record["after"]["fields"]["amount_rub"]["value"] == 200
    assert record["seed_proposal"] == proposal(100).model_dump(mode="json")


def test_saved_trace_loader_takes_last_checkpoint_and_checks_dialogue_before_paid_calls(tmp_path):
    from scripts.audit_jev import load_initial_seeds

    records = asyncio.run(audit_cases([case()], Extractor(), Jev()))
    path = tmp_path / "trace.jsonl"
    old = {**records[0], "before": None}
    path.write_text("\n".join(json.dumps(r) for r in [old, records[0]]))
    seeds = load_initial_seeds(path, [case()])
    assert seeds["C1"][0].document_type == "transfer_draft"
    assert seeds["C1"][1] == proposal(100)
    with pytest.raises(ValueError, match="диалог"):
        changed = case().model_copy(update={"messages": []})
        load_initial_seeds(path, [changed])
    with pytest.raises(ValueError, match="C2"):
        load_initial_seeds(path, [case("C2")])
    assert build_parser().parse_args(["--seed-trace", str(path)]).seed_trace == path


def test_saved_trace_loader_rejects_missing_seed_or_inconsistent_checked_fields(tmp_path):
    from scripts.audit_jev import load_initial_seeds

    records = asyncio.run(audit_cases([case()], Extractor(), Jev()))
    path = tmp_path / "trace.jsonl"
    record = records[0]
    record["before"]["fields"]["amount_rub"]["value"] = 999
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="кандидат"):
        load_initial_seeds(path, [case()])
    record["seed_proposal"] = None
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        load_initial_seeds(path, [case()])


def test_partial_seed_trace_requires_opt_in_and_keeps_the_original_candidate(tmp_path):
    from scripts.audit_jev import load_initial_seeds

    record = asyncio.run(audit_cases([case()], Extractor(), Jev()))[0]
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        load_initial_seeds(path, [case(), case("C2")])
    seeds = load_initial_seeds(path, [case(), case("C2")], allow_missing=True)
    assert list(seeds) == ["C1"]
    assert seeds["C1"][1].fields["amount_rub"].value == 100  # Не исправленный after=200.
    with pytest.raises(ValueError):
        load_initial_seeds(path, [case("C2")], allow_missing=True)
    assert build_parser().parse_args(["--allow-missing-seeds"]).allow_missing_seeds


def test_mixed_saved_and_fresh_seeds_reverify_every_candidate_without_old_verdicts():
    from src.pipeline import DocumentTypeDecision

    class MixedExtractor(Extractor):
        async def extract(self, *args, **kwargs):
            self.extractions.append(kwargs)
            return proposal(200 if kwargs.get("repair_fields") else 100)

    extractor, jev = MixedExtractor(), Jev()
    seeds = {"C1": (DocumentTypeDecision(document_type="transfer_draft", confidence=0.9), proposal(100))}
    records = asyncio.run(audit_cases([case(), case("C2")], extractor, jev, initial_seeds=seeds))
    assert extractor.classifications == 1
    assert len(jev.states) == 4  # Оба исходных ответа и оба ремонта проверены заново.
    assert all(r["before"]["fields"]["amount_rub"]["value"] == 100 for r in records)
    assert all(r["after"]["fields"]["amount_rub"]["value"] == 200 for r in records)


def test_partial_seed_trace_skips_failed_baseline_but_never_ignores_changed_dialogue(tmp_path):
    from scripts.audit_jev import load_initial_seeds

    complete = asyncio.run(audit_cases([case()], Extractor(), Jev()))[0]
    failed = {**complete, "case_id": "C2", "before": None, "status": "api_error"}
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in (complete, failed)))
    seeds = load_initial_seeds(path, [case(), case("C2")], allow_missing=True)
    assert list(seeds) == ["C1"]
    changed = case("C2").model_copy(update={"messages": []})
    with pytest.raises(ValueError, match="диалог"):
        load_initial_seeds(path, [case(), changed], allow_missing=True)


@pytest.mark.parametrize("timeout", [False, True])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("targeted", [False, True])
def test_saved_seed_cli_wires_recheck_and_writes_a_separate_report(tmp_path, monkeypatch, capsys, timeout, partial, targeted):
    from types import SimpleNamespace
    from src.llm import LLMUsage
    from src.settings import LLMSettings
    import scripts.audit_jev as module

    source = asyncio.run(audit_cases([case()], Extractor(), Jev()))[0]
    trace = tmp_path / "source.jsonl"
    original = json.dumps(source)
    trace.write_text(original)
    input_path, labels_path = tmp_path / "cases.jsonl", tmp_path / "labels.jsonl"
    selected_cases = [case(), case("C2")] if partial else [case()]
    input_path.write_text("\n".join(item.model_dump_json() for item in selected_cases))
    gold = label(100)
    gold.pop("scoring_only")  # Метаданные этого теста не входят в схему документа.
    labels_path.write_text("\n".join(json.dumps({**gold, "case_id": item.case_id}) for item in selected_cases))

    class SDK:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass

    class AcceptingVerifier:
        models = {"offline-test"}
        input_tokens = output_tokens = 0
        async def ainvoke(self, state, questions, response_model):
            if timeout:
                from typesafe_sdk import TypeSafeAPITimeoutError
                raise TypeSafeAPITimeoutError(30)
            return response_model.model_validate({"answers": {key: {"noul": 0.01} for key in questions.root}})

    class NoGeneration:
        usage = LLMUsage()
        def __init__(self, *args):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def classify(self, *args, **kwargs):
            assert partial
            return DocumentClassification(document_type="transfer_draft", confidence=0.9, none_reason=None)
        async def extract(self, *args, **kwargs):
            assert partial
            return proposal(100)

    settings = LLMSettings(_env_file=None, LLM_API_KEY="fake", LLM_MODEL="small")
    monkeypatch.setattr(module, "LLMSettings", lambda: settings)
    monkeypatch.setattr(module, "JevClientSettings", lambda: SimpleNamespace(api_key=settings.api_key))
    monkeypatch.setattr(module, "AsyncTypeSafeClient", SDK)
    monkeypatch.setattr(module, "AuditJev", lambda sdk: AcceptingVerifier())
    monkeypatch.setattr(module, "BudgetedExtractor", NoGeneration)
    output = tmp_path / "new-audit"
    args = build_parser().parse_args(["--live", "--input", str(input_path), "--labels", str(labels_path),
                                     "--seed-trace", str(trace), "--output-dir", str(output)] +
                                    (["--allow-missing-seeds"] if partial else []) +
                                    (["--case-id", "C1"] if targeted else []))
    assert asyncio.run(module.run_live(args)) is (not timeout)
    report = json.loads((output / "report.json").read_text())
    if timeout:
        assert report["conclusion"] == "incomplete"
        assert report["incomplete_cases"] == ["C1"]
        checkpoints = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
        assert checkpoints[-1]["status"] == "api_error"
    assert report["selected_cases"] == (["C1", "C2"] if partial and not targeted else ["C1"])
    assert report["baseline_origin"] == ("saved_and_fresh_trace" if partial and not targeted else "saved_trace")
    assert report["saved_seed_cases"] == ["C1"]
    assert report["fresh_seed_cases"] == (["C2"] if partial and not targeted else [])
    assert report["seed_trace_sha256"] and report["field_rules_sha256"]
    assert report["llm_http_calls"] == 0  # NoGeneration is an offline fake, not an HTTP client.
    assert report["jev_api_calls"] == (2 if partial and not targeted and not timeout else 1)
    assert "До Jev" in capsys.readouterr().out
    assert trace.read_text() == original


@pytest.mark.parametrize("kind", ["timeout", "connection"])
def test_sdk_network_oserrors_are_recorded_as_api_errors(kind):
    from typesafe_sdk import TypeSafeAPITimeoutError, TypeSafeAPIConnectionError

    class UnavailableJev:
        async def ainvoke(self, *args):
            if kind == "timeout":
                raise TypeSafeAPITimeoutError(30)
            raise TypeSafeAPIConnectionError("PRIVATE_SERVER_BODY")

    events = []
    records = asyncio.run(audit_cases(
        [case(), case("C2")], Extractor(), UnavailableJev(),
        on_event=lambda record: events.append(json.loads(json.dumps(record))),
    ))
    assert len(records) == 1
    assert records[0]["status"] == events[-1]["status"] == "api_error"
    assert "PRIVATE_SERVER_BODY" not in json.dumps(events)
    report = build_report(records, [label()])
    assert report["conclusion"] == "incomplete"
    assert report["incomplete_cases"] == ["C1"]


def test_audit_checkpoint_oserror_still_propagates():
    def checkpoint(record):
        if record["before"] is not None:
            raise OSError("disk failure")
    with pytest.raises(OSError, match="disk failure"):
        asyncio.run(audit_cases([case()], Extractor(), Jev(), on_event=checkpoint))
