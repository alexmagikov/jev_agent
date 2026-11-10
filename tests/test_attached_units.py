"""Примыкающие единицы не мешают источникам чисел и не разрешают обрезание."""
import pytest

from src.extraction import ExtractionError, get_proposal_model
from tests.test_extraction import proposal, filled, prepare


@pytest.mark.parametrize('text,value', [
    ('5000руб', 5000), ('5000руб.', 5000), ('5000рублей', 5000),
    ('5000Р', 5000), ('1 500,25руб', 1500.25), ('1\u202f500,25руб', 1500.25),
])
def test_direct_amount_with_attached_currency(text, value):
    data = proposal(amount_rub=filled(value, text))
    get_proposal_model('invoice_draft').model_validate(data)
    result = prepare(data, 'Счёт на ' + text)
    assert result.amount_rub.value == value


@pytest.mark.parametrize('text,count', [('3шт', 3), ('3штуки', 3), ('2,5кг', 2.5), ('2часа', 2)])
def test_item_quantity_with_attached_unit(text, count):
    data = proposal(items=filled([{
        'description': 'товар', 'quantity': count, 'unit': None, 'unit_price_rub': None,
    }], text))
    result = prepare(data, 'Счёт за ' + text + ' товара')
    assert result.items.value[0].quantity == count


def test_calculation_operands_allow_attached_units():
    text = 'Счёт за 3шт по 5000руб'
    data = proposal(amount_rub={**filled(None, text), 'calculation': {
        'expression': 'count * price', 'operands': [
            {'name': 'count', 'source': {'message_id': 'm1', 'quote': '3'}},
            {'name': 'price', 'source': {'message_id': 'm1', 'quote': '5000'}},
        ],
    }})
    assert prepare(data, text).amount_rub.value == 15000


@pytest.mark.parametrize('expression,operand', [('x', '200'), ('x', '1'), ('x', '5')])
def test_attached_unit_does_not_allow_number_fragments_as_operands(expression, operand):
    text = 'Счёт за 1200,5шт'
    data = proposal(amount_rub={**filled(None, text), 'calculation': {
        'expression': expression,
        'operands': [{'name': 'x', 'source': {'message_id': 'm1', 'quote': operand}}],
    }})
    with pytest.raises(ExtractionError):
        prepare(data, text)


@pytest.mark.parametrize('source,value', [
    ('A5000руб', 5000), ('5000abc', 5000), ('5000рубеж', 5000), ('3штрих', 3),
    ('12,5артикул', 12), ('12,5артикул', 5), ('1 000артикул', 1),
])
def test_unknown_word_or_identifier_is_not_a_measurement_suffix(source, value):
    with pytest.raises(ExtractionError):
        prepare(proposal(amount_rub=filled(value, source)), 'Код ' + source)


def test_direct_decimal_cannot_be_truncated_before_attached_unit():
    with pytest.raises(ExtractionError):
        prepare(proposal(amount_rub=filled(12, '12,5руб')), 'Счёт на 12,5руб')
