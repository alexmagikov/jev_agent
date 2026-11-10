"""Предрегистрированный анализ: бутстреп по кейсам, тест Макнемара, pass^k; без API."""

import asyncio
import json

import pytest

import scripts.analyze_experiment as analysis
from tests.test_experiment import experiment, http_models, prepared  # noqa: F401


@pytest.mark.parametrize("full_only,llm_only,p_value", [
    (4, 1, 0.375),
    (2, 0, 0.5),
    (8, 0, 0.0078125),
    (0, 0, 1.0),
])
def test_exact_mcnemar_uses_only_discordant_cases(full_only, llm_only, p_value):
    assert analysis.mcnemar_exact(full_only, llm_only) == pytest.approx(p_value)
    assert analysis.mcnemar_exact(llm_only, full_only) == pytest.approx(p_value)


def test_pass_hat_k_requires_all_rounds_to_succeed():
    outcomes = [[True, True, False], [True, True, True]]
    assert analysis.pass_hat_k(outcomes, 3) == pytest.approx(0.5)
    assert analysis.pass_hat_k(outcomes, 1) == pytest.approx((2 / 3 + 1) / 2)


def test_bootstrap_resamples_whole_cases_and_is_reproducible():
    assert analysis.bootstrap_difference([1, 0, 1], [1, 0, 1]) == (0.0, 0.0, 0.0)
    assert analysis.bootstrap_difference([1, 1, 1], [0, 0, 0]) == (1.0, 1.0, 1.0)
    full, llm = [1, 1, 0, 1, 2 / 3], [0, 1, 0, 1, 1 / 3]
    first = analysis.bootstrap_difference(full, llm)
    assert first == analysis.bootstrap_difference(full, llm)
    estimate, low, high = first
    assert estimate == pytest.approx((1 + 0 + 0 + 0 + 1 / 3) / 5)
    assert low <= estimate <= high


@pytest.mark.parametrize("low,high,verdict", [
    (0.05, 0.3, "full лучше"), (-0.3, -0.05, "llm-only лучше"), (-0.1, 0.2, "разница не установлена"),
])
def test_verdict_requires_the_whole_interval_on_one_side(low, high, verdict):
    assert analysis.verdict(low, high) == verdict


def test_analysis_reads_saved_runs_and_writes_report(prepared, http_models):
    prepared.live, prepared.rounds = True, 2
    report_path = asyncio.run(experiment.run_experiment(prepared))
    result = analysis.analyze(report_path.parent)
    assert result["cases"] == 1 and result["rounds"] == 2 and result["status"] == "completed"
    # В фикстуре full исправляет сумму, llm-only оставляет устаревшее значение.
    assert result["accuracy"] == {"llm-only": 0.0, "full": 1.0}
    assert result["difference"] == {"estimate": 1.0, "ci95": [1.0, 1.0], "verdict": "full лучше"}
    assert result["mcnemar_round1"] == {"full_only": 1, "llm_only_only": 0, "p_value": 1.0}
    assert result["pass_hat_k"] == {"k": 2, "llm-only": 0.0, "full": 1.0}
    saved = json.loads((report_path.parent / "analysis.json").read_text(encoding="utf-8"))
    assert saved == result


def test_silent_errors_count_wrong_documents_accepted_without_review(prepared, http_models):
    prepared.live, prepared.rounds = True, 2
    report_path = asyncio.run(experiment.run_experiment(prepared))
    result = analysis.analyze(report_path.parent)
    # llm-only принимает устаревшую сумму без пометки; full её исправляет.
    assert result["silent_errors"]["rate"] == {"llm-only": 1.0, "full": 0.0}
    assert result["silent_errors"]["difference"] == {"estimate": -1.0, "ci95": [-1.0, -1.0], "verdict": "full лучше"}


def test_subgroups_use_challenge_families_from_tags(prepared, http_models, tmp_path):
    prepared.live, prepared.rounds = True, 1
    report_path = asyncio.run(experiment.run_experiment(prepared))
    case_id = next(iter(analysis.case_outcomes(report_path.parent)[1]))
    manifest = tmp_path / "challenge.manifest.json"
    manifest.write_text(json.dumps({"cases": [{"case_id": case_id, "tags": ["reverted_correction", "arithmetic"]}]}))
    result = analysis.analyze(report_path.parent, manifest)
    assert result["subgroups_descriptive"]["reverted_correction"]["cases"] == 1
