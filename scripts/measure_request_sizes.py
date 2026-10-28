"""Размеры запросов без API: реальный сборщик LLM + сериализатор TypeSafe SDK.

Запуск из корня проекта: python -m scripts.measure_request_sizes
--export-dir сохраняет тела запросов и учебные примеры для отчёта.
Токены здесь не оцениваются; ответы модели не генерируются.
"""

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from statistics import median

import httpx2
from typesafe_sdk._core.config import Config
from typesafe_sdk._core.endpoints import prepare_system_one
from typesafe_sdk.constants import DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_TIMEOUT

from src.cases.document_type import DocumentMessage, DocumentTypeCase, DocumentTypeState
from src.extraction import ExtractionProposal, get_proposal_model, get_template, prepare_fields
from src.llm import AsyncLLMExtractor, build_extraction_prompt
from src.settings import LLMSettings
from src.verification import (
    VerificationResponse, VerificationState, _ITEMS_RULES, _RULES,
    build_verification_questions, verify_fields,
)


ROOT = Path(__file__).resolve().parents[1]


def dump(value):
    return json.dumps(value, ensure_ascii=False)


def size(value):
    text = value.decode("utf-8") if isinstance(value, bytes) else value
    return {"characters": len(text), "utf8_bytes": len(text.encode("utf-8"))}


def c001_proposal():
    """Учебный корректный ответ, составленный вручную, а не результат нового LLM-запроса."""
    fields = {
        name: {"status": "missing", "value": None, "evidence": [], "calculation": None}
        for name in get_template("invoice_draft")["fields"]
    }
    values = {
        "recipient_name": ("ООО Ромашка", "ООО Ромашка"),
        "recipient_inn": ("7701234567", "ИНН 7701234567"),
        "subject": ("консультационные услуги", "консультационные услуги"),
        "items": ([{
            "description": "консультационная услуга", "quantity": None, "unit": None,
            "unit_price_rub": None, "quantity_calculation": None, "unit_price_calculation": None,
        }], "консультационные услуги"),
        "amount_rub": (5000, "5000 рублей"),
        "vat_mode": ("no_vat", "без НДС"),
    }
    for name, (value, quote) in values.items():
        fields[name].update(status="fillable", value=value, evidence=[{"message_id": "m1", "quote": quote}])
    get_proposal_model("invoice_draft").model_validate({"fields": fields})
    return ExtractionProposal.model_validate({"fields": fields})


def jev_body(state, questions, response_model):
    # Явные учебные настройки: не читаем .env и не вызываем сеть.
    config = Config("offline", DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_TIMEOUT, httpx2.Headers())
    request = prepare_system_one(
        config, state.model_dump(mode="json", by_alias=True),
        questions.model_dump(mode="json", by_alias=True),
        None, None, None, None, response_model,
    )
    return request.content


async def main(export_dir=None):
    cases = [json.loads(line) for line in (ROOT / "data/cases.jsonl").read_text().splitlines() if line.strip()]
    # Только для офлайн-разбивки размеров по типам; labels не входят в запросы.
    types = {record["case_id"]: record["document_type"]
             for line in (ROOT / "data/labels.jsonl").read_text().splitlines() if line.strip()
             for record in [json.loads(line)]}
    c001 = next(case for case in cases if case["case_id"] == "C001")
    messages = [DocumentMessage.model_validate(message) for message in c001["messages"]]
    proposal = c001_proposal()
    checked = prepare_fields(proposal, messages, "invoice_draft")
    questions = build_verification_questions(checked, proposal=proposal, document_type="invoice_draft", messages=messages)
    state = VerificationState(
        messages=messages, document_type="invoice_draft", rules=_RULES,
    )
    classification = DocumentTypeCase(state=DocumentTypeState(messages=messages))
    examples = {
        "01-jev-classification.request.json": jev_body(classification.state, classification.questions, classification.response_model),
        "03-llm-response.example.json": dump(proposal.model_dump(mode="json")).encode(),
        "04-python-fields.example.json": dump(checked.model_dump(mode="json")).encode(),
        "05-jev-verification.request.json": jev_body(state, questions, VerificationResponse),
    }

    class ExampleJev:
        async def ainvoke(self, *, state, questions, response_model):
            return response_model.model_validate({"answers": {
                name: {"noul": 0.89 if name == "items::unsupported" else 0.01}
                for name in questions.root
            }})

    jev_feedback = await verify_fields(messages, "invoice_draft", proposal, checked, ExampleJev())
    python_feedback = ["Цитата источника отсутствует в указанной реплике"]
    captured = []

    def respond(request):
        captured.append(request.content)
        request_user = json.loads(json.loads(request.content)["messages"][1]["content"])
        # Это заглушка только для завершения локального extract(), не предсказание.
        body = {"fields": {name: {"status": "missing", "value": None, "evidence": [], "calculation": None}
                           for name in request_user["template"]["fields"]}}
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": dump(body)}}]})

    settings = LLMSettings(
        _env_file=None, LLM_API_KEY="offline", LLM_MODEL="qwen2.5:7b",
        LLM_BASE_URL="http://127.0.0.1:11434/v1", LLM_TIMEOUT_SECONDS=180,
        LLM_MAX_OUTPUT_TOKENS=4096, LLM_JSON_MODE=True, LLM_RESPONSE_FORMAT="json_schema",
        LLM_TEMPERATURE=0, LLM_TOKEN_LIMIT_PARAMETER="max_tokens", LLM_MAX_RETRIES=1,
        LLM_INPUT_PRICE_PER_MILLION=0, LLM_OUTPUT_PRICE_PER_MILLION=0,
    )
    groups = defaultdict(list)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        async with AsyncLLMExtractor(settings, client=client) as extractor:
            for name, feedback in (
                ("02-llm-extraction.request.json", []),
                ("06-llm-retry-python.request.json", python_feedback),
                ("07-llm-retry-jev.request.json", jev_feedback),
            ):
                repair = {
                    "previous_proposal": proposal, "previous_checked_fields": checked,
                    "repair_fields": ["items"], "repair_paths": ["items"],
                } if name == "07-llm-retry-jev.request.json" else {}
                await extractor.extract(messages, "invoice_draft", feedback=feedback, **repair)
                examples[name] = captured[-1]
            for case in cases:
                kind = types[case["case_id"]]
                if kind == "none":
                    continue
                await extractor.extract([DocumentMessage.model_validate(m) for m in case["messages"]], kind)
                payload = json.loads(captured[-1])
                groups[(case["split"], kind)].append({
                    "http_body_bytes": len(captured[-1]),
                    "message_content_characters": sum(len(m["content"]) for m in payload["messages"]),
                })

    llm = json.loads(examples["02-llm-extraction.request.json"])
    user = json.loads(llm["messages"][1]["content"])
    assert user["output_schema"] == llm["response_format"]["json_schema"]["schema"]
    assert user["messages"] == [m.model_dump() for m in messages]
    assert not {"case_id", "labels", "split"} & user.keys()
    for filename in ("06-llm-retry-python.request.json", "07-llm-retry-jev.request.json"):
        retry = json.loads(examples[filename])
        retry_user = json.loads(retry["messages"][1]["content"])
        assert retry_user.pop("feedback")
        if filename == "07-llm-retry-jev.request.json":
            assert retry_user.pop("previous_proposal") == proposal.model_dump(mode="json")
            assert retry_user.pop("previous_checked_fields") == checked.model_dump(mode="json")
            assert retry_user.pop("python_changes") == {}
            assert retry_user.pop("repair_fields") == ["items"]
            assert retry_user.pop("repair_paths") == ["items"]
            assert retry_user["template"]["fields"] == {"items": user["template"]["fields"]["items"]}
            repair_schema = retry_user["output_schema"]
            repair_fields_schema = repair_schema["$defs"][repair_schema["properties"]["fields"]["$ref"].split("/")[-1]]
            assert set(repair_fields_schema["properties"]) == set(repair_fields_schema["required"]) == {"items"}
            assert retry["response_format"]["json_schema"]["schema"] == repair_schema
            # Остальной контекст тот же; схема и смысл полей намеренно сужены до ремонта.
            retry_user["template"] = user["template"]
            retry_user["output_schema"] = user["output_schema"]
        assert retry_user == {k: v for k, v in user.items() if k != "feedback"}
    question_counts = Counter(name.split("::")[1] for name in questions.root)
    counts = Counter((case["split"], types[case["case_id"]]) for case in cases)
    stats = {
        "method": "offline: actual LLM request via MockTransport; TypeSafe SDK request builder; no API calls",
        "typesafe_sdk_version": version("typesafe-sdk"),
        "tokens": None,
        "dataset": [{"split": split, "document_type": kind, "cases": count}
                    for (split, kind), count in sorted(counts.items())],
        "system_prompt": size(build_extraction_prompt("invoice_draft")),
        "system_prompts": {kind: size(build_extraction_prompt(kind))
                           for kind in ("invoice_draft", "transfer_draft")},
        "c001": {
            "repair_output_fields": ["items"],
            "user_content": size(llm["messages"][1]["content"]),
            "system_plus_user_content": size("".join(m["content"] for m in llm["messages"])),
            "user_components": {key: size(dump(value)) for key, value in user.items()},
            "request_and_example_sizes": {name: size(body) for name, body in examples.items()},
            "classification_questions": size(dump(classification.questions.model_dump(mode="json"))),
            "verification_state_components": {key: size(dump(value)) for key, value in state.model_dump(mode="json").items()},
            "verification_questions": size(dump(questions.model_dump(mode="json"))),
            "verification_question_count": len(questions.root),
            "verification_question_kinds": dict(question_counts),
            "base_rules_in_state": size(_RULES),
            "base_rules_total_characters": len(_RULES),
            "item_rules_per_question": size(_ITEMS_RULES),
            "item_rules_copies": sum(_ITEMS_RULES in question.instructions["rules"] for question in questions.root.values()),
            "python_feedback": size(dump(python_feedback)),
            "jev_feedback": size(dump(jev_feedback)),
        },
        "first_llm_requests_by_expected_type": [{
            "split": split, "document_type": kind, "cases": len(rows),
            **{metric: {"min": min(row[metric] for row in rows),
                        "median": median(row[metric] for row in rows),
                        "max": max(row[metric] for row in rows),
                        "total": sum(row[metric] for row in rows)}
               for metric in ("http_body_bytes", "message_content_characters")},
        } for (split, kind), rows in sorted(groups.items())],
    }
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)
        for name, body in examples.items():
            (export_dir / name).write_bytes(body)
        (export_dir / "sizes.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, help="Каталог генерируемых примеров и sizes.json")
    args = parser.parse_args()
    asyncio.run(main(args.export_dir))
