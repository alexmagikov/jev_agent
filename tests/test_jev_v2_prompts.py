"""Запросы сохраняют снимок jev-v2 вне явно разрешённых исправлений.

Снимок уже содержит уточнение единственного числа и исполняется только как эталон
сборки; Python-валидация остаётся текущей. Качество ответов модели здесь не проверяется.
"""

import builtins
import json
from pathlib import Path
import sys
import types

import pytest

from scripts.measure_request_sizes import jev_body
from src.cases.document_type import DocumentMessage, DocumentTypeCase, DocumentTypeState
from src.extraction import ExtractionProposal
from src.field_rules import ITEM_UNIT_RULE, ITEM_DESCRIPTION_SEMANTIC_RULE, QUOTED_PRODUCT_NAME_RULE, TRANSFER_ALIAS_RULES, TRANSFER_RECIPIENT_RULE, INVOICE_AMOUNT_RULE, BASIS_DATE_RULE, DESTINATION_BANK_RULE, RECIPIENT_NAME_RULE
from src.field_rules import INVOICE_SUBJECT_RULE, ITEMS_EXAMPLES, VAT_MODE_RULE
from src.schemas.document_schemas import InvoiceFields, TransferFields
from src.verification import VerificationState, VerificationResponse, _RULES, build_verification_questions


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "prompts/versions/jev-v2"
CASES = {row["case_id"]: row for line in (ROOT / "data/cases_v2.jsonl").read_text().splitlines()
         if (row := json.loads(line))}
LABELS = {row["case_id"]: row for line in (ROOT / "data/labels_v2.jsonl").read_text().splitlines()
          if (row := json.loads(line))["document_type"] != "none"}

@pytest.fixture
def archived_verifier(monkeypatch):
    rules = types.ModuleType("_test_v1_rules")
    exec(compile((SNAPSHOT / "field_rules.py.txt").read_text(), "v1/field_rules.py", "exec"), rules.__dict__)
    module = types.ModuleType("_test_v1_verification")
    original_import = builtins.__import__

    def archive_import(name, *args, **kwargs):
        return rules if name == "src.field_rules" else original_import(name, *args, **kwargs)

    module.__dict__["__builtins__"] = {**vars(builtins), "__import__": archive_import}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile((SNAPSHOT / "verification.py.txt").read_text(), "v1/verification.py", "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("case_id", LABELS)
def test_verification_wire_matches_snapshot_except_authorized_semantic_fixes(case_id, archived_verifier):
    # Labels supply varied fixture statuses only; no labels are put in request state.
    label = LABELS[case_id]
    messages = [DocumentMessage.model_validate(row) for row in CASES[case_id]["messages"]]
    source = {message.id: message.text for message in messages}
    model = InvoiceFields if label["document_type"] == "invoice_draft" else TransferFields
    fields = model.model_validate(label["fields"])
    proposal = ExtractionProposal.model_validate({"fields": {
        name: {"status": field["status"], "value": field["value"],
               "evidence": [{"message_id": mid, "quote": source[mid]} for mid in field["evidence"]]}
        for name, field in label["fields"].items()
    }})
    kwargs = {"proposal": proposal, "document_type": label["document_type"], "messages": messages}
    current_questions = build_verification_questions(fields, **kwargs)
    old_questions = archived_verifier.build_verification_questions(fields, **kwargs)
    current_state = VerificationState(messages=messages, document_type=label["document_type"], rules=_RULES)
    old_state = archived_verifier.VerificationState(
        messages=messages, document_type=label["document_type"], rules=archived_verifier._RULES,
    )
    expected = json.loads(
        jev_body(old_state, old_questions, archived_verifier.VerificationResponse)
    )
    # Python проверяет предложенное значение; для пропущенного передаётся правило.
    expected["state"]["rules"] = expected["state"]["rules"].replace(
        "Numeric formats, quote membership and arithmetic are already checked by Python.",
        "Python checks formats of proposed values, quote membership and calculation arithmetic.\n"
        "    It has not checked every value in messages. For a status question, use\n"
        "    instructions.field.format_check: when only a rule is supplied, apply it to relevant user data.\n"
        "    matches_format is a Python fact about input_value, not about ownership or current applicability.",
    )
    expected["state"]["rules"] = expected["state"]["rules"].replace(
        '    Evaluate instructions.field using its meaning, cited evidence and the entire dialogue in messages.\n',
        '    Evaluate instructions.field using its meaning, cited evidence and the entire dialogue in messages.\n'
        '    instructions.field.path is a logical field label, not a JSON lookup path.\n'
        '    Read the checked value from instructions.field.value and its sources from instructions.field.evidence.\n'
        '    Calculation plans, when supplied, are in instructions.field.calculation or instructions.field.item_calculations.\n',
    )
    for key, question in list(expected["questions"].items()):
        claim = question["instructions"]["field"]
        # ФИО при неоднозначном падеже сохраняется в форме пользователя.
        question["instructions"]["rules"] = [
            RECIPIENT_NAME_RULE if rule == archived_verifier.RECIPIENT_NAME_RULE else rule
            for rule in question["instructions"]["rules"]
        ]
        # Авторизованное уточнение: форма единицы количества имеет приоритет над формой цены.
        question["instructions"]["rules"] = [
            ITEM_UNIT_RULE if rule == archived_verifier.ITEM_UNIT_RULE else rule
            for rule in question["instructions"]["rules"]
        ]
        # Различаем отдельные платы за позиции и одну плату за обе вместе.
        question["instructions"]["rules"] = [
            INVOICE_AMOUNT_RULE if rule == archived_verifier.INVOICE_AMOUNT_RULE else rule
            for rule in question["instructions"]["rules"]
        ]
        # Эквивалентное написание больше не означает безусловное подтверждение значения.
        for name, rule in TRANSFER_ALIAS_RULES.items():
            question["instructions"]["rules"] = [
                rule if old == archived_verifier.TRANSFER_ALIAS_RULES[name] else old
                for old in question["instructions"]["rules"]
            ]
        # Ревью промптов: падеж subject, пример с subject, «С НДС» без ставки.
        for old, new in ((archived_verifier.INVOICE_SUBJECT_RULE, INVOICE_SUBJECT_RULE),
                         (archived_verifier.ITEMS_EXAMPLES, ITEMS_EXAMPLES),
                         (archived_verifier.VAT_MODE_RULE, VAT_MODE_RULE)):
            question["instructions"]["rules"] = [rule.replace(old, new) for rule in question["instructions"]["rules"]]
        # ambiguous включает неясный смысл или принадлежность данных, как у извлекателя.
        if key.endswith("::status_wrong"):
            question["instructions"]["question"] = question["instructions"]["question"].replace(
                "alternatives between usable values; fillable",
                "alternatives between usable values or unclear meaning or ownership of supplied data; fillable",
            )
        # Разрешённая нормализация названий теперь учитывается во всех проверках названий.
        question["instructions"]["rules"] = [
            rule.replace(archived_verifier.ITEM_DESCRIPTION_SEMANTIC_RULE, ITEM_DESCRIPTION_SEMANTIC_RULE)
            for rule in question["instructions"]["rules"]
        ]
        path = claim["path"]
        if path == "basis_date":
            question["instructions"]["rules"].append(BASIS_DATE_RULE)
        # Проверки связи получателя теперь получают общее с извлекателем правило self.
        if path == "recipient_relation":
            question["instructions"]["rules"].append(TRANSFER_RECIPIENT_RULE)
        # Банк перевода приводится к именительному падежу; правило общее с извлекателем.
        if path == "destination_bank":
            question["instructions"]["rules"].append(DESTINATION_BANK_RULE)
        if path == "subject" or path == "items" or path.endswith("].description"):
            if QUOTED_PRODUCT_NAME_RULE not in question["instructions"]["rules"]:
                question["instructions"]["rules"].append(QUOTED_PRODUCT_NAME_RULE)
        if not key.endswith(("::missing_supported", "::unresolved_alternatives")):
            claim.pop("valid_source_candidates", None)
        if key.endswith("::unsupported"):
            question["instructions"]["question"] = (
                f"Is instructions.field.value for `{claim['path']}` unsupported by its cited user messages "
                "in instructions.field.evidence? Use instructions.field.meaning and the supplied rules. "
                "Support may be semantic, use permitted normalization, or follow a justified calculation. "
                "Check instructions.field.calculation or instructions.field.item_calculations when supplied; "
                "judge the source operands and operation. The result need not appear verbatim in the messages."
            )
            question["criteria"] = {
                "true": (
                    "The cited user information does not support instructions.field.value under "
                    "instructions.field.meaning and the supplied rules: neither direct semantic support, "
                    "permitted normalization, nor justified source operands and a calculation support it."
                ),
                "false": (
                    "The cited user information supports instructions.field.value in meaning, by permitted "
                    "normalization, or by a justified calculation. The result need not appear verbatim in "
                    "the messages; this does not relax the verbatim requirement for evidence quotes."
                ),
            }
        if key.endswith("::noncanonical"):
            question["instructions"]["question"] = (
                f"Does instructions.field.value for `{claim['path']}` violate the explicit canonical-format rule "
                "supplied in instructions.rules for this field? Judge only that rule; permitted "
                "normalization and other valid forms are not formatting errors."
            )
            question["criteria"] = {
                "true": (
                    "instructions.field.value violates an explicit canonical-format rule supplied in "
                    "instructions.rules for this field."
                ),
                "false": (
                    "instructions.field.value follows the explicit canonical-format rule, including "
                    "permitted normalization and other valid forms. A different valid form alone is "
                    "not an error; support and ownership are judged separately."
                ),
            }

        if key == "items::incomplete":
            question["instructions"]["question"] = (
                "Does `items` omit any active, unambiguously selected whole line item "
                "requested by the user? Apply later corrections and withdrawals. "
                "Judge only missing line items; do not assess attributes of items already "
                "present, including description, quantity, unit or price. Check missing "
                "attributes only in their separate attribute questions. Unchosen alternatives "
                "and withdrawn items are not missing line items."
            )
            question["criteria"] = {
                "true": (
                    "An active, unambiguously selected whole line item requested by the user "
                    "is absent from items after later corrections and withdrawals."
                ),
                "false": (
                    "No active, unambiguously selected whole line item is missing. Missing or "
                    "incorrect attributes of items already present belong to separate attribute "
                    "checks. Unchosen alternatives and withdrawn items are not omissions."
                ),
            }

        if key.endswith("::wrong_role") and claim["status"] == "fillable":
            question["instructions"]["question"] = (
                f"Does the information in instructions.field.evidence used to justify "
                f"instructions.field.value for `{claim['path']}` belong to a different entity or numerical "
                "role than required by instructions.field.meaning? Use messages to identify "
                "the owner and role of that specific information, including source operands "
                "of a calculated value. Other entities' details elsewhere in the dialogue or "
                "in the same quote do not make a correctly attributed value wrong. "
                "Flag an actual owner or role mismatch; format errors or missing support alone "
                "are not this error."
            )
            if label["document_type"] == "invoice_draft" and claim["path"].startswith("recipient_"):
                question["instructions"]["question"] += (
                    " For this invoice, the expected recipient is the buyer/customer being billed; "
                    "account and bank details must refer to that buyer's account or bank."
                )

            question["criteria"] = {
                "true": (
                    "The specific cited information or source operands used to justify "
                    "instructions.field.value belong to a different entity or numerical role "
                    "than required by instructions.field.meaning."
                ),
                "false": (
                    "The information used to justify the value concerns the intended entity and role. "
                    "Other entities' details, including a seller's details beside the correct buyer's "
                    "details in the same quote, do not count. Format errors or missing support alone "
                    "are not an owner or role mismatch."
                ),
            }

        if key.endswith("::wrong_role") and claim["status"] != "fillable":
            question["instructions"]["question"] = (
                f"Do the cited sources in instructions.field.evidence for `{claim['path']}` support information "
                "belonging to a different person, recipient, bank, document or numerical role than this field? "
                "Use messages to resolve the owner and role of the cited information. "
                "Other entities' details elsewhere in the dialogue do not make these citations wrong. "
                "A malformed value for the correct recipient is not this error."
            )
            question["criteria"] = {
                "true": "The information in instructions.field.evidence belongs to a different entity or role than this field.",
                "false": "The cited information concerns the intended entity and role, even if malformed; unrelated details elsewhere in messages do not count.",
            }
        if not key.endswith("::superseded"):
            continue
        if claim["status"] == "missing":
            del expected["questions"][key]
        elif claim["status"] != "fillable":
            question["instructions"]["question"] = (
                f"Does the current null status in instructions.field.status for `{claim['path']}` "
                "conflict with the latest explicit user correction for this field? "
                "Judge the final status, not whether the cited information changed. "
                "Withdrawal without replacement requires missing/null. "
                "If the latest data still justify invalid/null or ambiguous/null, answer false. "
                "Corrections to other fields and malformed data alone do not count."
            )
            question["criteria"] = {
                "true": "The current null status conflicts with the latest explicit user correction for this field.",
                "false": "The current null status is still justified, or no correction applies; a historical source change alone is not an error.",
            }
    actual = json.loads(jev_body(current_state, current_questions, VerificationResponse))
    # Новые проверки статуса и описаний покрыты отдельными контрактными тестами.
    # Здесь сохраняется точное сравнение всех остальных вопросов со снимком v1.
    actual["questions"] = {key: question for key, question in actual["questions"].items()
                           if not key.endswith(("::status_wrong", ".description::value_wrong",
                                                ".description::unsupported_characteristics",
                                                ".description::noncanonical"))}
    expected["questions"] = {key: question for key, question in expected["questions"].items()
                             # Общие проверки формул заменены вопросами по исходным планам.
                             if not key.endswith((".description::value_wrong", "::formula_wrong"))}
    assert actual == expected


def test_classification_and_template_match_v1_except_recipient_correction_policy():
    resolved = json.loads((SNAPSHOT / "resolved_rules.json").read_text())
    case = DocumentTypeCase(state=DocumentTypeState(messages=[
        DocumentMessage(id="m1", role="user", text="Выставь счёт"),
    ]))
    actual = case.questions.model_dump(mode="json")
    expected = resolved["classification_questions"]
    # Новый порядок применения исправлений проверяется в
    # test_recipient_correction_classification.py. Остальной контракт v1 сохранён.
    for question, section, fields in (
        ("document_type", "instructions", ("request_rule", "precedence_rule")),
        ("document_type", "criteria", ("transfer_draft",)),
        ("none_reason", "criteria", ("other_document",)),
    ):
        for field in fields:
            expected[question][section][field] = actual[question][section][field]
    expected["none_reason"]["instructions"] = actual["none_reason"]["instructions"]
    assert actual == expected
    assert json.loads((ROOT / "data/form_templates.json").read_text()) == json.loads(
        (SNAPSHOT / "form_templates.json").read_text()
    )
