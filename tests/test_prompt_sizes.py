"""Лимиты фактического контекста: символы, без оценки токенов или качества моделей."""
import json
from pathlib import Path

from scripts.measure_request_sizes import c001_proposal
from src.cases.document_type import DocumentMessage
from src.extraction import prepare_fields
from src.field_rules import QUOTED_PRODUCT_NAME_RULE
from src.llm import build_extraction_prompt
from src.verification import build_verification_questions


def test_invoice_extraction_prompt_stays_under_character_budget():
    # Полное правило даты добавило 687 символов: текущий промпт — 13184.
    # Округление до копеек и запасной вариант для ФИО с примером: текущий промпт — 13322.
    # Ревью промптов (покупатель, длины счёта и БИК, падеж subject, НДС, константы): 13804.
    # Услуги и работы во множественном числе без количества сохраняются: 14128.
    # Итог по строкам q1 * p1 + q2 * p2 и цитата возвращённого значения (challenge dev): 14313.
    assert len(build_extraction_prompt("invoice_draft")) <= 14400


def test_c001_verification_questions_stay_under_character_budget():
    path = Path(__file__).resolve().parents[1] / "data/cases.jsonl"
    case = next(row for line in path.read_text().splitlines()
                if (row := json.loads(line))["case_id"] == "C001")
    messages = [DocumentMessage.model_validate(message) for message in case["messages"]]
    proposal = c001_proposal()
    prepared = prepare_fields(proposal, messages, "invoice_draft")
    questions = build_verification_questions(prepared, proposal=proposal, messages=messages)
    status_checks = {key: question for key, question in questions.root.items() if key.endswith("::status_wrong")}
    specific_checks = {key: question for key, question in questions.root.items() if not key.endswith("::status_wrong")}
    # Явные пути и критерии wrong_role/incomplete/unsupported/noncanonical расширили вопросы.
    # Общее правило даты добавлено к вопросам basis_date.
    # Измеренные размеры C001: 70748 / 74728 / 77296 символов для трёх групп ниже.
    # Фиксированные пределы оставляют небольшой запас; status_wrong проверяется отдельно.
    # Общая нормализация названий учитывается отдельно от остальных правил.
    # Запасной вариант для ФИО повторяется в вопросах recipient_name: 71083 / 75063 / 77631.
    # Ревью промптов: падеж subject в правилах позиций, пример subject, «С НДС»: 73257 / 77237 / 79805.
    # Проверка формы описания позиции (description::noncanonical): 73257 / 78661 / 81443.
    # Правило для услуг без количества в этой проверке: 73257 / 79089 / 81871.
    base_checks = {key: question.model_dump(mode="json") for key, question in specific_checks.items()}
    for question in base_checks.values():
        question["instructions"]["rules"] = [rule for rule in question["instructions"]["rules"]
                                               if rule != QUOTED_PRODUCT_NAME_RULE]
    other_checks = {key: question for key, question in base_checks.items() if ".description::" not in key}
    assert len(type(questions).model_validate(other_checks).model_dump_json()) <= 73400
    assert len(type(questions).model_validate(base_checks).model_dump_json()) <= 79200
    assert len(type(questions).model_validate(specific_checks).model_dump_json()) <= 82000
    assert len(status_checks) == len(type(prepared).model_fields)
    # Правило даты получает и basis_date::status_wrong; текущая группа — 20516 символов.
    # Определение ambiguous с неясной принадлежностью данных во всех status_wrong: 21673.
    assert len(type(questions).model_validate(status_checks).model_dump_json()) <= 21800
