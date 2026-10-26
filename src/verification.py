"""Модель решений проверяет смысл каждого поля против диалога, включая пропуски."""

from typing import Literal

from pydantic import BaseModel, JsonValue, RootModel

from src.cases.document_type import DocumentMessage
from src.client import DecisionModelClient
from src.extraction import (
    ExtractionProposal, PreparedFields, SupportedDocumentType, get_template, requisite_format_context,
    requisite_source_candidates,
)
from src.field_rules import (
    INVOICE_AMOUNT_RULE, INVOICE_SUBJECT_RULE, ITEM_DESCRIPTION_SEMANTIC_RULE, ITEM_QUANTITY_RULE,
    ITEM_UNIT_PRICE_RULE, ITEM_UNIT_RULE, TRANSFER_ALIAS_RULES,
    ITEMS_PRESENCE_RULE, ITEMS_SELECTION_RULE, ITEMS_EXAMPLES, RECIPIENT_NAME_RULE, BASIS_NUMBER_RULE,
    VAT_MODE_RULE, VAT_RATE_RULE, QUOTED_PRODUCT_NAME_RULE,
)
from src.schemas.schemas import Probability


class VerificationError(RuntimeError):
    """Ответ верификатора не содержит ожидаемый набор проверок."""


class NoulQuestion(BaseModel):
    type: Literal["noul"] = "noul"
    instructions: dict[str, JsonValue]
    criteria: dict[Literal["true", "false"], str]


class NoulAnswer(BaseModel):
    noul: Probability


class VerificationResponse(BaseModel):
    answers: dict[str, NoulAnswer]


class VerificationState(BaseModel):
    messages: list[DocumentMessage]
    document_type: SupportedDocumentType
    rules: str


class VerificationQuestions(RootModel[dict[str, NoulQuestion]]):
    pass


_RULES = (
    """
    Your task is to verify whether the document fields are correctly filled
    according to the selected document template and the user's dialogue.
    For each question, answer instructions.question; a high probability means that specific error is present.
    Evaluate instructions.field using its meaning, cited evidence and the entire dialogue in messages.
    Use only information provided by the user. Assistant messages may clarify what a user’s reply
    refers to, but cannot independently support a value.
    A later explicit correction supersedes the previous value.
    Numeric formats, quote membership and arithmetic are already checked by Python.
    Apply the explicit field formatting rules below; permitted grammatical normalization
    does not make a value unsupported just because it is not a verbatim substring.
    For a directly stated value, check that its sources support it.
    For a calculated value, check that the source numbers were selected correctly
    and that the formula matches the meaning of the message. The calculated result
    does not have to appear verbatim in the text.
    Return the probability that the specified error is present."""
    " Item descriptions are normalized to singular nominative when unambiguous ('стулья' -> 'стул'). "
    "This preserves product identity, including when the source is plural; it is not an unsupported "
    "value or a different product. Preserve quoted names, plural-only nouns and ambiguous forms. "
    "Do not infer quantity=1 or change units/evidence."
)


_ITEMS_RULES = " ".join((
    ITEMS_PRESENCE_RULE,
    ITEMS_SELECTION_RULE,
    INVOICE_SUBJECT_RULE,
    ITEM_DESCRIPTION_SEMANTIC_RULE,
    ITEM_QUANTITY_RULE,
    ITEM_UNIT_PRICE_RULE,
    ITEMS_EXAMPLES,
))


_FIELD_RULES = {
    "recipient_name": RECIPIENT_NAME_RULE,
    "basis_number": BASIS_NUMBER_RULE,
    **TRANSFER_ALIAS_RULES,
    "vat_mode": VAT_MODE_RULE,
    "vat_rate_percent": VAT_RATE_RULE,
}


_STATUS_RULE = (
    "Python owns format validation. Do not count digits, check registries/checksums, or pad values. "
    "Malformed source data alone is not an extraction error: invalid/null is correct for it. "
    "valid_source_candidates, when present, contains only format-valid user quotes, not approved "
    "recipient details. Judge their ownership, numerical role, latest corrections and unresolved "
    "alternatives from messages. A seller's or sender's details do not fill the recipient's field. "
    "Without an unambiguous supported usable value, null is correct even for a required field. "
    "The question judges one semantic error, not whether the document is ready to issue."
)


_SEMANTIC_CRITERIA = {
    "status_wrong": {
        "true": "The current status is wrong: the latest applicable user data require a different status among fillable, missing, invalid and ambiguous.",
        "false": "The status is justified. A wrong value or citation under the same correct status belongs to another check.",
    },
    "missing_supported": {
        "true": "A current unambiguous usable value for this field was omitted.",
        "false": "No current unambiguous usable value is omitted; null is justified.",
    },
    "wrong_role": {
        "true": "The information in instructions.field.evidence belongs to a different entity or role than this field.",
        "false": "The cited information concerns the intended entity and role, even if malformed; unrelated details elsewhere in messages do not count.",
    },
    "superseded": {
        "true": "The current null status conflicts with the latest explicit user correction for this field.",
        "false": "The current null status is still justified, or no correction applies; a historical source change alone is not an error.",
    },
    "unresolved_alternatives": {
        "true": "Actual mutually exclusive usable alternatives remain unresolved; ambiguity was not represented.",
        "false": "No unresolved choice is misrepresented; absence or bad format alone is not ambiguity.",
    },
}


def _build_verification_checks(fields: PreparedFields) -> dict[str, str]:
    """Конкретные вопросы проверок без общих правил верификатора."""

    questions = {}
    for name, field in fields.model_dump(mode="json").items():
        path = f"`{name}`"
        questions[f"{name}::status_wrong"] = (
            f"Does the current status in instructions.field.status for {path} contradict the latest "
            "applicable user data? Judge status only, not value correctness. missing means no current "
            "applicable data; invalid means supplied but unusable data; ambiguous means unresolved "
            "alternatives between usable values; fillable means supported usable data, including "
            "a justified calculation. Explicit malformed data is not missing. Respect ownership, "
            "withdrawals, corrections and the supplied field-specific rules."
        )
        
        if field["status"] == "fillable":
            checks = {
                "unsupported": f"Is the value of {path} unsupported by its cited user messages?",
                "wrong_role": f"Does {path} refer to the wrong person, item, bank, document or numerical role?",
                "superseded": (
                    f"Does the proposed value in instructions.field.value for {path} conflict with "
                    "the latest explicit user correction for this field, or remain filled after "
                    "explicit withdrawal? Judge the candidate, not whether any correction occurred. "
                    "If the candidate already reflects the latest correction, answer false. "
                    "Corrections to other fields do not count."
                ),
            }

            if name == "items":
                alternative_question = (
                    "Does `items` include an unchosen item or guess an attribute despite unresolved, "
                    "mutually exclusive alternatives in the user's latest applicable messages? "
                )
                expected_state = (
                    "The expected item selection/status is described in the supplied item-selection rules."
                )
            else:
                alternative_question = (
                    f"Does {path} contain a definite value despite unresolved, mutually exclusive "
                    "alternatives in the user's latest applicable messages? "
                )
                expected_state = (
                    "The expected state is status=ambiguous, value=null, with evidence of the alternatives."
                )
                
            checks["unresolved_alternatives"] = (
                alternative_question
                + "Resolved choices, permitted equivalent forms, jointly requested values, "
                "missing information alone and model uncertainty are not this error. "
                + expected_state
            )

            if name == "amount_rub" and "items" in type(fields).model_fields:
                checks["ambiguous"] = (
                    f"Does {path} incorrectly claim a definite invoice total when the dialogue "
                    "leaves unit price versus line total unresolved? The presence of this number "
                    "in a source does not establish its numerical role. Apply the shared invoice-request "
                    "convention first: a single unqualified amount attached to the invoice request "
                    "supports the total unless context indicates otherwise; missing quantity alone "
                    "does not make that total ambiguous. If still unresolved, the field "
                    "must have status=ambiguous, value=null and evidence of the ambiguous wording."
                )

            if name in ("recipient_name", "basis_number", "subject"):
                checks["noncanonical"] = (
                    f"Does {path}.value violate the explicit canonical-format rule for this field? "
                    "Flag only an unambiguous formatting error, not a merely different valid name."
                )

            if name == "items":
                checks["incomplete"] = (
                    "Does `items` omit an active item or explicitly supplied part of an item? "
                    "Only flag parts supplied by the user or unambiguously computable from their data; "
                    "unknown attributes correctly left null are not omissions."
                )
            
            if field["derivation"]:
                calculation_location = (
                    "Check `quantity_calculation` and `unit_price_calculation` inside each item "
                    "in instructions.field.item_calculations."
                    if name == "items"
                    else "Check instructions.field.calculation."
                )

                checks["formula_wrong"] = (
                    f"Does the calculation for {path} misinterpret quantities, unit prices, totals, "
                    "discount scope, units or requested operations? " + calculation_location
                )
        else:
            checks = {
                "missing_supported": (
                    f"Does the latest user dialogue supply an unambiguous usable value for {path} "
                    "that this null field omits, directly stated or unambiguously computable? "
                    "Require support for this field's meaning and recipient/entity, "
                    "not merely any mentioned value. Unresolved alternatives, withdrawn values and "
                    "malformed source data alone are not a usable omitted value."
                ),
            }
            
            if field["evidence"]:
                checks["wrong_role"] = (
                    f"Do the cited sources in instructions.field.evidence for {path} support information "
                    "belonging to a different person, recipient, bank, document or numerical role than this field? "
                    "Use messages to resolve the owner and role of the cited information. "
                    "Other entities' details elsewhere in the dialogue do not make these citations wrong. "
                    "A malformed value for the correct recipient is not this error."
                )
                # Missing after withdrawal is correct; an omitted replacement is
                # already covered by missing_supported. Only a stale invalid or
                # ambiguous status needs this additional temporal check.
                if field["status"] != "missing":
                    checks["superseded"] = (
                        f"Does the current null status in instructions.field.status for {path} "
                        "conflict with the latest explicit user correction for this field? "
                        "Judge the final status, not whether the cited information changed. "
                        "Withdrawal without replacement requires missing/null. "
                        "If the latest data still justify invalid/null or ambiguous/null, answer false. "
                        "Corrections to other fields and malformed data alone do not count."
                    )
                
            if field["status"] != "ambiguous":
                checks["unresolved_alternatives"] = (
                    f"Does the latest user dialogue leave mutually exclusive usable alternatives "
                    f"for {path} unresolved, requiring ambiguous/null rather than its current status? "
                    "Require actual unchosen alternatives, not absent data, malformed data alone, "
                    "equivalent forms, or model uncertainty."
                )
                
            if name == "subject" and field["status"] == "missing":
                checks["missing_supported"] = (
                    f"Is {path} wrongly left missing although the user explicitly names a "
                    "single billed goods/service category for the current invoice? "
                    "Check the actual billed category in the user's messages, not whether "
                    "a separate purpose sentence exists. Unknown quantity or an unresolved "
                    "price does not remove a named subject. Do not flag this omission if "
                    "only a recipient, amount, contract, or several different goods without "
                    "a stated common purpose are supplied."
                )

            if name == "vat_rate_percent":
                checks["missing_supported"] = (
                    "Does this null field omit an unambiguous, explicitly stated current VAT "
                    "percentage for this invoice, including an explicit 0%? Do not infer a "
                    "percentage from 'без НДС' (no_vat), tax/base amounts, or calculations. "
                    "Ignore superseded values, quoted instructions, and unrelated percentages."
                )
                
            if name == "amount_rub" and field["status"] == "missing" and "items" in type(fields).model_fields:
                checks["missing_supported"] = (
                    f"Is {path} wrongly left missing although an invoice total is explicitly "
                    "stated or an unambiguous sum of all active billed line charges can be "
                    "computed from the user's messages? Check each named line and the role "
                    "of its price. An explicitly assigned charge per named line can support "
                    "the sum even with unknown quantity within that line; a mere unit price "
                    "with unknown quantity cannot. For goods, 'по ... рублей каждый' gives "
                    "unit prices, not whole-line charges. Without a separately stated total, "
                    "missing is correct if any active line lacks both a whole-line charge "
                    "and enough data to calculate it. Never sum unit prices once per product name "
                    "or treat a known subtotal as the invoice total. "
                    "A computed total need not appear verbatim "
                    "in the source. Do not flag if the total requires invented quantities "
                    "or leaves the scope of a charge unresolved."
                )

        for check, question in checks.items():
            questions[f"{name}::{check}"] = question

        if name == "items" and field["status"] == "fillable":
            for index, item in enumerate(field["value"]):
                for attribute, value in item.items():
                    path = f"`items[{index}].{attribute}`"
                    if value is None:
                        questions[f"items[{index}].{attribute}::missing_supported"] = (
                            f"Is {path} wrongly left null although user messages unambiguously "
                            "supply this attribute or support its calculation for this single line item? "
                            "Check the attribute's numerical role and association with this item, "
                            "including later corrections. If the attribute is not uniquely supported, "
                            "null is correct. Do not fill it merely because other attributes are known."
                        )
                        continue
                    if attribute == "description":
                        questions[f"items[{index}].{attribute}::value_wrong"] = (
                            f"Does {path} name a different product/service or add unsupported "
                            "characteristics for this single line item compared with the latest user request? "
                            "Singular nominative normalization preserves identity; "
                            "ignore meaning-preserving number/case changes; "
                            "do not judge quantity, unit or price here."
                        )
                        continue

                    questions[f"items[{index}].{attribute}::value_wrong"] = (
                        f"Is {path} incorrect for this single line item? "
                        "Check this attribute individually: its meaning, association with this item, "
                        "latest user correction, and source support. "
                        "Correct attributes of the same or other items do not justify an incorrect one."
                    )
                    if attribute in ("quantity", "unit_price_rub"):
                        questions[f"items[{index}].{attribute}::value_wrong"] += (
                            " For a calculated number, judge the source operands and operation, "
                            "not verbatim result membership."
                        )

    return questions


def build_verification_questions(
    fields: PreparedFields, *, proposal: ExtractionProposal | None = None,
    document_type: SupportedDocumentType | None = None,
    messages: list[DocumentMessage] | None = None,
) -> VerificationQuestions:
    """Собирает вопросы для Jev по подготовленным полям и исходному ответу модели.

    proposal содержит извлечённые значения, цитаты и планы расчётов; messages —
    диалог с пользователем. Для полей без значения Python подбирает цитаты,
    прошедшие проверку формата.
    """

    questions = {}
    document_type = document_type or ("invoice_draft" if "items" in type(fields).model_fields else "transfer_draft")
    template = get_template(document_type)
    prepared = fields.model_dump(mode="json")
    
    attribute_rules = {"description": ITEM_DESCRIPTION_SEMANTIC_RULE, "quantity": ITEM_QUANTITY_RULE,
                       "unit": ITEM_UNIT_RULE, "unit_price_rub": ITEM_UNIT_PRICE_RULE}
    attribute_meanings = {"description": "Наименование этой позиции счёта",
                          "quantity": "Количество товара/услуги в этой позиции счёта",
                          "unit": "Единица измерения количества этой позиции счёта",
                          "unit_price_rub": "Цена за одну единицу этой позиции счёта в рублях, не сумма строки"}
    
    candidates = {
        name: requisite_source_candidates(name, messages)
        for name, field in prepared.items()
        if messages is not None and field["status"] != "fillable"
    }

    for key, question in _build_verification_checks(fields).items():
        name = key.split("::", 1)[0].split("[", 1)[0]
        source_candidates = candidates.get(name)
        if source_candidates is not None:
            if key.endswith("::missing_supported") and not source_candidates:
                continue 
            if key.endswith("::unresolved_alternatives") and len({c["quote"] for c in source_candidates}) < 2:
                continue

        item_rules = _ITEMS_RULES if name == "items" else ""
        semantic_rule = ""
        if "items" in type(fields).model_fields:
            semantic_rule = {"amount_rub": INVOICE_AMOUNT_RULE,
                             "subject": INVOICE_SUBJECT_RULE, "items": ITEM_UNIT_RULE}.get(name, "")
        if key.startswith("items["):
            attribute = key.split("::", 1)[0].rsplit(".", 1)[-1]
            item_rules = attribute_rules[attribute]
            semantic_rule = ""

        path = key.split("::", 1)[0]
        current = prepared[name]
        raw = proposal.fields[name].model_dump(mode="json") if proposal is not None else None
        claim = {
            "path": path, "meaning": template["fields"][name],
            "status": current["status"], "value": current["value"],
            "evidence": raw["evidence"] if raw is not None else [{"message_id": mid} for mid in current["evidence"]],
            "calculation": raw["calculation"] if raw is not None else None,
            "derivation": current["derivation"],
        }

        format_check = requisite_format_context(name, raw["value"] if raw is not None else None)
        if format_check is not None and (current["status"] == "fillable" or (
                key.endswith("::status_wrong") and "matches_format" in format_check)):
            claim["format_check"] = format_check
        
        if source_candidates is not None and key.endswith(("::missing_supported", "::unresolved_alternatives")):
            claim["valid_source_candidates"] = source_candidates
        
        if path.startswith("items["):
            index = int(path.split("[", 1)[1].split("]", 1)[0])
            attribute = path.rsplit(".", 1)[1]
            claim["meaning"] = attribute_meanings[attribute]
            claim["item"] = current["value"][index]
            claim["value"] = claim["item"][attribute]
            calculation_name = {"quantity": "quantity_calculation", "unit_price_rub": "unit_price_calculation"}.get(attribute)
            claim["calculation"] = raw["value"][index].get(calculation_name) if raw is not None and calculation_name else None
            claim.pop("derivation")
        elif name == "items" and raw is not None and isinstance(raw["value"], list):
            claim["item_calculations"] = [
                {key: item[key] for key in ("quantity_calculation", "unit_price_calculation")}
                for item in raw["value"]
            ]

        criteria = _SEMANTIC_CRITERIA[key.split("::", 1)[1]] if current["status"] != "fillable" or key.endswith("::status_wrong") else {
            "true": "Yes. The specific error in instructions.question is present.",
            "false": "No. The specific error is absent; unrelated errors do not count.",
        }

        if path.endswith("].unit_price_rub") and key.endswith("::value_wrong") and claim["calculation"] is not None:
            question = (
                f"Do the cited operands or operation for `{path}` contradict the latest user "
                "conditions for this line item's unit price? Check which line and numerical role "
                "each operand belongs to and whether a later correction replaces it. "
                "Python checked arithmetic; judge operand selection and the requested operation."
            )
            criteria = {
                "true": "An operand is unsupported, superseded or from a different line/role, or the operation contradicts the requested unit price.",
                "false": "The operands and operation follow the latest applicable conditions. A batch total divided by its positive count supports this line's unit price unless different prices are stated. A historical price alone does not invalidate current operands; the result need not appear verbatim.",
            }
        
        if path.endswith("].description") and key.endswith("::value_wrong"):
            criteria = {
                "true": "The description names a different product/service or adds unsupported characteristics.",
                "false": "Same requested product/service, including singular nominative normalization; meaning-preserving number/case changes and unknown quantity/unit/price are not description errors.",
            }
        if key.endswith("::superseded") and current["status"] == "fillable":
            criteria = {
                "true": "The latest explicit user correction requires a different value or withdraws this field's value.",
                "false": "The candidate already reflects the latest correction, or none applies; a historical correction alone is not an error.",
            }
        if key == "vat_rate_percent::missing_supported":
            criteria = {
                "true": "An unambiguous explicit current VAT percentage, including 0%, was omitted.",
                "false": "No explicit current VAT percentage is omitted; no_vat alone and calculated rates do not count.",
            }

        quoted_name_rule = QUOTED_PRODUCT_NAME_RULE if key == "subject::noncanonical" else ""
        questions[key] = NoulQuestion(
            instructions={"question": question, "field": claim, "rules": [part for part in
                          (_FIELD_RULES.get(name), VAT_MODE_RULE if name == "vat_rate_percent" else "",
                           item_rules, semantic_rule, quoted_name_rule,
                           _STATUS_RULE if format_check is not None and current["status"] != "fillable"
                           and not key.endswith("::status_wrong") else "") if part]},
            criteria=criteria,
        )
        
    return VerificationQuestions(questions)


async def verify_fields(
    messages: list[DocumentMessage],
    document_type: SupportedDocumentType,
    proposal: ExtractionProposal,
    fields: PreparedFields,
    client: DecisionModelClient,
    wrong_threshold: float = 0.8,
    *,
    debug: bool = False,
) -> list[str]:
    if not 0 < wrong_threshold < 1:
        raise ValueError("Порог проверки должен находиться между 0 и 1")

    questions = build_verification_questions(fields, proposal=proposal, document_type=document_type, messages=messages)
    response = await client.ainvoke(
        state=VerificationState(
            messages=messages,
            document_type=document_type,
            rules=_RULES,
        ),
        questions=questions,
        response_model=VerificationResponse,
    )

    if set(response.answers) != set(questions.root):
        raise VerificationError(f"{getattr(client, 'name', 'Decision model')} вернул неполный или неожиданный набор проверок")

    if debug:
        print(f"{getattr(client, 'name', 'Decision model')}: оценки ошибок, порог={wrong_threshold:.2f}:", flush=True)
        for key, answer in response.answers.items():
            print(f"  {key}: {answer.noul:.3f}", flush=True)

    return [
        f"{key}: {getattr(client, 'name', 'Decision model')} обнаружил возможную ошибку ({answer.noul:.2f}). "
        + questions.root[key].instructions["question"]
        for key, answer in response.answers.items()
        if answer.noul >= wrong_threshold
    ]
