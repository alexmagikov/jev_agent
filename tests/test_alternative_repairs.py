"""Контракт вопросов и ремонта вариантов; не оценка качества реального Jev."""
import asyncio

import pytest

from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, prepare_fields
from src.field_rules import ITEMS_SELECTION_RULE
from src.llm import build_extraction_prompt
from src.pipeline import DocumentTypeDecision, process_selected_document
from src.schemas.document_schemas import InvoiceFields
from src.verification import build_verification_questions
from tests.test_verification import _inputs


def test_scalar_question_does_not_include_item_instructions():
    _, _, fields = _inputs()
    text = build_verification_questions(fields).root['amount_rub::unresolved_alternatives'].instructions['question']
    assert 'items' not in text
    assert 'latest applicable' in text
    assert 'status=ambiguous' in text and 'value=null' in text
    assert 'expected' in text.lower()
    assert 'use status=' not in text
    for exception in ('Resolved choices', 'equivalent forms', 'jointly requested', 'missing information', 'model uncertainty'):
        assert exception in text
    assert len(text) < 500


@pytest.mark.parametrize('dialogue,initial_items,repaired_items', [
    ('Стол и либо стул, либо кресло — ещё решу.', ['стол', 'стул'], ['стол']),
    ('Либо стул, либо кресло — ещё решу.', ['стул'], None),
    ('Стул, либо 2, либо 3 штуки — ещё решу.', ['стул'], ['стул']),
])
def test_item_alternatives_repair_preserves_confirmed_items_and_other_fields(dialogue, initial_items, repaired_items):
    messages = [DocumentMessage(id='m1', role='user', text='Выставь счёт: ' + dialogue)]
    fields = {name: {'status': 'missing', 'value': None, 'evidence': []}
              for name in InvoiceFields.model_fields}
    def item(name, quantity=None):
        return {'description': name, 'quantity': quantity, 'unit': None, 'unit_price_rub': None}
    fields['items'] = {'status': 'fillable',
        'value': [item(name, 2 if '2, либо 3' in dialogue else None) for name in initial_items],
        'evidence': [{'message_id': 'm1', 'quote': dialogue}]}
    initial = ExtractionProposal.model_validate({'fields': fields})
    before = prepare_fields(initial, messages, 'invoice_draft')
    calls = []
    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            calls.append(questions.root)
            if len(calls) == 1:
                question = questions.root['items::unresolved_alternatives']
                assert 'status=ambiguous' not in question.instructions['question']
                assert ITEMS_SELECTION_RULE in question.instructions['rules'][0]
                assert 'unchosen' in question.instructions['question']
            return response_model.model_validate({'answers': {
                key: {'noul': .95 if len(calls) == 1 and key == 'items::unresolved_alternatives' else .01}
                for key in questions.root
            }})
    class Extractor:
        async def extract(self, *args, **kwargs):
            assert kwargs['repair_paths'] == ['items']
            assert kwargs['repair_fields'] == ['items']
            assert len(kwargs['feedback']) == 1
            repaired = initial.model_dump()
            repaired['fields']['items'] = {
                'status': 'ambiguous' if repaired_items is None else 'fillable',
                'value': None if repaired_items is None else [item(name) for name in repaired_items],
                'evidence': [{'message_id': 'm1', 'quote': dialogue}],
            }
            return ExtractionProposal.model_validate(repaired)
    assert ITEMS_SELECTION_RULE in build_extraction_prompt('invoice_draft')
    assert ITEMS_SELECTION_RULE not in build_extraction_prompt('transfer_draft')
    result = asyncio.run(process_selected_document('alternatives', messages,
        DocumentTypeDecision(document_type='invoice_draft', confidence=.9),
        Verifier(), Extractor(), initial_proposal=initial))
    assert len(calls) == 2
    actual = result.fields.items
    assert actual.status == ('ambiguous' if repaired_items is None else 'fillable')
    if repaired_items is None:
        assert actual.value is None
    else:
        assert [i.description for i in actual.value] == repaired_items
        assert all(i.quantity is None for i in actual.value)
    for name in InvoiceFields.model_fields:
        if name != 'items':
            assert getattr(result.fields, name) == getattr(before, name)
