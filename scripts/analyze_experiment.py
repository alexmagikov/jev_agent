"""Предрегистрированный анализ эксперимента (docs/experiment-protocol.md), без запросов к API.

Единица анализа — кейс. Для каждого кейса и режима берётся доля раундов с полностью
правильным документом. Кейс со сбоем API хотя бы в одном режиме или раунде исключается
из сравнения парно; прочие необработанные кейсы считаются ошибкой. Главная оценка —
разница full − llm-only с 95% интервалом парного бутстрепа по кейсам. Дополнительно:
точный тест Макнемара по первому раунду и pass^k по всем раундам.

Запуск: python -m scripts.analyze_experiment experiments/<id>
"""

import argparse
import json
import random
from math import comb
from pathlib import Path

from src.metrics import exact_document
from src.run_artifacts import load_run

MODES = ("llm-only", "full")
SEED = 20261028
RESAMPLES = 10_000
ROOT = Path(__file__).resolve().parents[1]
MAX_EXCLUDED_SHARE = 0.10  # больше исключённых — сравнение не предъявляется как итоговое


def mcnemar_exact(full_only: int, llm_only_only: int) -> float:
    """Двусторонний точный тест: при равенстве режимов расхождения делятся 50/50."""
    discordant = full_only + llm_only_only
    if not discordant:
        return 1.0
    tail = sum(comb(discordant, i) for i in range(min(full_only, llm_only_only) + 1)) / 2 ** discordant
    return min(1.0, 2 * tail)


def pass_hat_k(outcomes: list[list[bool]], k: int) -> float:
    """Вероятность, что все k раундов кейса успешны, в среднем по кейсам (τ-bench)."""
    return sum(comb(sum(rounds), k) / comb(len(rounds), k) for rounds in outcomes) / len(outcomes)


def bootstrap_difference(full: list[float], llm_only: list[float], *, resamples: int = RESAMPLES,
                         seed: int = SEED) -> tuple[float, float, float]:
    """Средняя разница долей по кейсам и перцентильный 95% интервал; кейс выбирается целиком."""
    differences = [a - b for a, b in zip(full, llm_only, strict=True)]
    size = len(differences)
    rng = random.Random(seed)
    means = sorted(sum(differences[i] for i in rng.choices(range(size), k=size)) / size
                   for _ in range(resamples))
    quantile = lambda share: means[round(share * (resamples - 1))]
    return sum(differences) / size, quantile(0.025), quantile(0.975)


def verdict(low: float, high: float) -> str:
    if low > 0:
        return "full лучше"
    if high < 0:
        return "llm-only лучше"
    return "разница не установлена"


def _run_directory(experiment: Path, phase: dict) -> Path:
    saved = Path(phase["run_directory"])
    if saved.exists():
        return saved
    # Каталог эксперимента могли перенести: ищем тот же запуск по относительному пути.
    return experiment / f"round-{phase['round']:03d}" / phase["mode"] / "predictions.runs" / saved.name


def case_outcomes(experiment: Path) -> tuple[dict, dict, dict, set]:
    report = json.loads((experiment / "report.json").read_text(encoding="utf-8"))
    config = json.loads((experiment / "config.json").read_text(encoding="utf-8"))
    labels = {row["case_id"]: row for row in (json.loads(line) for line in
              (experiment / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    outcomes = {mode: {case_id: [False] * config["rounds"] for case_id in labels} for mode in MODES}
    api_errors = set()
    for phase in report["phases"]:
        if phase["mode"] in MODES and phase.get("run_directory"):
            bundle = load_run(_run_directory(experiment, phase))
            for row in bundle["predictions"]:
                outcomes[phase["mode"]][row["case_id"]][phase["round"] - 1] = exact_document(labels[row["case_id"]], row)
            api_errors |= {error["case_id"] for error in bundle["errors"] if error.get("kind") == "api_error"}
    return outcomes, labels, report, api_errors


def _difference(outcomes: dict, case_ids: list[str]) -> dict:
    shares = {mode: [sum(outcomes[mode][cid]) / len(outcomes[mode][cid]) for cid in case_ids] for mode in MODES}
    estimate, low, high = bootstrap_difference(shares["full"], shares["llm-only"])
    return {"estimate": estimate, "ci95": [low, high], "verdict": verdict(low, high)}


def analyze(experiment: Path, manifest: Path | None = None) -> dict:
    outcomes, labels, report, api_errors = case_outcomes(experiment)
    excluded = sorted(api_errors)
    case_ids = [case_id for case_id in labels if case_id not in api_errors]
    if not case_ids:
        raise ValueError("Все кейсы исключены из-за сбоев API: сравнивать нечего")
    rounds = len(next(iter(outcomes["full"].values())))
    full_first = {cid: outcomes["full"][cid][0] for cid in case_ids}
    llm_first = {cid: outcomes["llm-only"][cid][0] for cid in case_ids}
    full_only = sum(full_first[cid] and not llm_first[cid] for cid in case_ids)
    llm_only_only = sum(llm_first[cid] and not full_first[cid] for cid in case_ids)
    result = {
        "status": report["status"], "cases": len(case_ids), "rounds": rounds, "excluded_api_error": excluded,
        "accuracy": {mode: sum(sum(outcomes[mode][cid]) for cid in case_ids) / (len(case_ids) * rounds) for mode in MODES},
        "difference": _difference(outcomes, case_ids),
        "mcnemar_round1": {"full_only": full_only, "llm_only_only": llm_only_only,
                           "p_value": mcnemar_exact(full_only, llm_only_only)},
        "pass_hat_k": {"k": rounds, **{mode: pass_hat_k([outcomes[mode][cid] for cid in case_ids], rounds)
                                       for mode in MODES}},
        "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "unit": "case"},
    }
    # Подгруппы — только описательно, без выводов о значимости.
    groups = {}
    manifest = manifest or ROOT / "data" / "evaluation50.manifest.json"
    if manifest.exists():
        for item in json.loads(manifest.read_text(encoding="utf-8"))["cases"]:
            if item["case_id"] in case_ids:
                groups.setdefault(item["group"], []).append(item["case_id"])
    for cid in case_ids:
        groups.setdefault(labels[cid]["document_type"], []).append(cid)
    result["subgroups_descriptive"] = {name: {"cases": len(ids), **_difference(outcomes, ids)}
                                       for name, ids in groups.items()}
    warnings = []
    if report["status"] != "completed":
        warnings.append("Эксперимент не завершён: по протоколу его нельзя предъявлять как итоговое сравнение.")
    if len(excluded) > MAX_EXCLUDED_SHARE * len(labels):
        warnings.append(f"Из-за сбоев API исключено {len(excluded)} из {len(labels)} кейсов (больше 10%): "
                        "по протоколу сравнение не предъявляется как итоговое.")
    if warnings:
        result["warning"] = " ".join(warnings)
    (experiment / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path, help="Каталог experiments/<id> с report.json")
    parser.add_argument("--manifest", type=Path, help="Манифест с группами кейсов (по умолчанию evaluation50)")
    args = parser.parse_args(argv)
    result = analyze(args.experiment, args.manifest)
    diff = result["difference"]
    print(f"Кейсов: {result['cases']}, раундов: {result['rounds']}, статус эксперимента: {result['status']}")
    if result["excluded_api_error"]:
        print(f"Исключены из-за сбоев API (парно): {', '.join(result['excluded_api_error'])}")
    print("Полностью правильные документы: " + ", ".join(f"{m} {result['accuracy'][m]:.1%}" for m in MODES))
    print(f"Разница full − llm-only: {diff['estimate']:+.1%}, 95% ДИ [{diff['ci95'][0]:+.1%}; {diff['ci95'][1]:+.1%}]"
          f" → {diff['verdict']}")
    mc = result["mcnemar_round1"]
    print(f"Макнемар (раунд 1): только full верно — {mc['full_only']}, только llm-only — {mc['llm_only_only']}, "
          f"p = {mc['p_value']:.3f}")
    ph = result["pass_hat_k"]
    print(f"pass^{ph['k']}: llm-only {ph['llm-only']:.1%}, full {ph['full']:.1%}")
    for name, group in result["subgroups_descriptive"].items():
        print(f"  {name} ({group['cases']}): {group['estimate']:+.1%} [{group['ci95'][0]:+.1%}; {group['ci95'][1]:+.1%}]")
    if "warning" in result:
        print(result["warning"])
    print(f"Сохранено: {args.experiment / 'analysis.json'}")


if __name__ == "__main__":
    main()
