"""Сложная выборка challenge: тонкие ошибки, на которых LLM должна ошибаться чаще.

Сценарии взяты из промахов финала на evaluation50 и теста Jev на внедрённых ошибках:
отсылки к ранее названным значениям, поправки через несколько реплик и их отмена,
чужие реквизиты, скидка на часть позиций, суммы в тысячах, отказ от документа.
Реквизиты вымышлены, но с верными контрольными разрядами ИНН и ключом счёта по БИК.

Первые 10 кейсов — split=dev для пилота трудности и работы с промптами.
Кейсы и эталоны составлены ассистентом и требуют проверки человеком; генерация
и --check не обращаются к API.
"""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.build_dataset_v2 import example, fill, item
from src.cases.document_type import NONE_REASON_TEXT
from src.extraction import invalid_format, validate_messages
from src.main import InputCase, LABEL_ADAPTER

DATA = Path(__file__).resolve().parents[1] / "data"


def scenario(case_id, kind, note, messages, *, tags, split="dev", **fields):
    case, label = example(case_id, split, kind, messages, **fields)
    return case, label, {"case_id": case_id, "split": split, "document_type": kind,
                         "tags": list(tags), "rationale": note}


def manual_records():
    """H001–H010, написаны вручную."""
    return [
        scenario("H001", "invoice_draft", "Отсылка «по той же цене» после смены количества: 55 × 85 + 12 × 30 = 5035.", [
            ("user", "Выставь счёт ООО Гранит-Сервис за кабель: 40 м по 85 рублей за метр. Назначение: Монтаж сети. Без НДС."),
            ("assistant", "Что-нибудь добавить?"),
            ("user", "Добавь коннекторы — 12 шт. по 30 рублей."),
            ("assistant", "Добавил коннекторы."),
            ("user", "Кабеля нужно не 40, а 55 м, по той же цене."),
        ], tags=("reference_to_earlier_value", "quantity_correction", "arithmetic"),
            recipient_name=fill("ООО Гранит-Сервис", "m1"), subject=fill("Монтаж сети", "m1"),
            items=fill([item("кабель", 55, "м", 85), item("коннектор", 12, "шт.", 30)], "m1", "m3", "m5"),
            amount_rub=fill(5035, "m1", "m3", "m5", derivation="55 * 85 + 12 * 30 = 5035 RUB"),
            vat_mode=fill("no_vat", "m1")),

        scenario("H002", "invoice_draft", "Поправка количества отменена отсылкой «как в самом начале»; цена стола изменена: 6 × 3200 + 2 × 9000 = 37200.", [
            ("user", "Подготовь счёт ООО Тихая гавань: стулья — 6 шт. по 3200 рублей, столы — 2 шт. по 9500 рублей. "
             "Назначение: Мебель для переговорной. Без НДС."),
            ("assistant", "Всё верно?"),
            ("user", "Стульев сделай 8."),
            ("assistant", "Изменил количество стульев на 8."),
            ("user", "Хотя нет, директор вернул как было: стульев столько, сколько я написал в самом начале. "
             "А столы по 9 000 рублей — скидку согласовали."),
        ], tags=("reverted_correction", "reference_to_earlier_value", "price_correction", "arithmetic"),
            recipient_name=fill("ООО Тихая гавань", "m1"), subject=fill("Мебель для переговорной", "m1"),
            items=fill([item("стул", 6, "шт.", 3200), item("стол", 2, "шт.", 9000)], "m1", "m5"),
            amount_rub=fill(37200, "m1", "m5", derivation="6 * 3200 + 2 * 9000 = 37200 RUB"),
            vat_mode=fill("no_vat", "m1")),

        scenario("H003", "invoice_draft", "Полные реквизиты продавца названы первыми и не относятся к покупателю.", [
            ("user", "Нужен счёт на оплату за аудит отчётности, 45 000 рублей, без НДС. Для шапки наши реквизиты: "
             "ООО Ревизор, ИНН 7705123452, р/с 40702810200000012345, БИК 044525225."),
            ("assistant", "Кто покупатель?"),
            ("user", "Покупатель — АО Северный порт: ИНН 7810543216, счёт 40702810200000067890, БИК 044030653, "
             "почта finance@sevport.example."),
        ], tags=("ownership", "seller_requisites_decoy", "complete_details"),
            recipient_name=fill("АО Северный порт", "m3"), recipient_inn=fill("7810543216", "m3"),
            recipient_account=fill("40702810200000067890", "m3"), recipient_bik=fill("044030653", "m3"),
            recipient_email=fill("finance@sevport.example", "m3"), subject=fill("аудит отчётности", "m1"),
            items=fill([item("аудит отчётности")], "m1"), amount_rub=fill(45000, "m1"), vat_mode=fill("no_vat", "m1")),

        scenario("H004", "invoice_draft", "Скидка 10% только на палатки и только в итоге: 3 × 12400 × 0,9 + 5 × 3100 = 48980.", [
            ("user", "Счёт ООО Лесная тропа: палатки — 3 шт. по 12 400 рублей, спальники — 5 шт. по 3 100 рублей. "
             "Назначение: Снаряжение для похода. Без НДС."),
            ("assistant", "Есть скидки?"),
            ("user", "На палатки скидка 10% — отрази её в итоге, цены в строках не меняй. Спальники без скидки."),
        ], tags=("partial_discount", "arithmetic", "multiple_items"),
            recipient_name=fill("ООО Лесная тропа", "m1"), subject=fill("Снаряжение для похода", "m1"),
            items=fill([item("палатка", 3, "шт.", 12400), item("спальник", 5, "шт.", 3100)], "m1", "m3"),
            amount_rub=fill(48980, "m1", "m3", derivation="3 * 12400 * (100 - 10) / 100 + 5 * 3100 = 48980 RUB"),
            vat_mode=fill("no_vat", "m1")),

        scenario("H005", "transfer_draft", "Сумма в тысячах с поправкой: 22,5 тыс. = 22,5 × 1000 = 22500; прежние 27,5 тыс. устарели.", [
            ("user", "С расчётного счёта переведи себе 27,5 тыс. рублей на карту ****4410 в Альфа-Банке."),
            ("assistant", "Подтверждаете 27 500 рублей?"),
            ("user", "Ой, нет: 22,5 тыс., а не 27,5."),
        ], tags=("thousands_shorthand", "latest_correction", "assistant_not_evidence"),
            amount_rub=fill(22500, "m3", derivation="22.5 * 1000 = 22500 RUB"),
            destination_bank=fill("Альфа-Банк", "m1"), destination_card_masked=fill("****4410", "m1"),
            source_account_kind=fill("расчётный счёт", "m1"), recipient_relation=fill("self", "m1")),

        scenario("H006", "invoice_draft", "Позиция снята, другая добавлена позже: 500 × 4 + 300 × 18 = 7400.", [
            ("user", "Счёт ООО Белый лист за печать: визитки — 500 шт. по 4 рубля, листовки — 2000 шт. по 3 рубля. "
             "Назначение: Полиграфия к выставке. Без НДС."),
            ("assistant", "Всё в одном счёте?"),
            ("user", "Листовки убери, их напечатали в другой типографии."),
            ("assistant", "Убрал листовки."),
            ("user", "Зато добавь буклеты — 300 шт. по 18 рублей."),
        ], tags=("withdrawn_item", "added_item", "arithmetic"),
            recipient_name=fill("ООО Белый лист", "m1"), subject=fill("Полиграфия к выставке", "m1"),
            items=fill([item("визитка", 500, "шт.", 4), item("буклет", 300, "шт.", 18)], "m1", "m3", "m5"),
            amount_rub=fill(7400, "m1", "m3", "m5", derivation="500 * 4 + 300 * 18 = 7400 RUB"),
            vat_mode=fill("no_vat", "m1")),

        scenario("H007", "transfer_draft", "Карта выбрана через отрицание «это не Сбер»: ВТБ ****7710.", [
            ("user", "Переведи себе 15 000 рублей с расчётного счёта. Карты у меня две: Сбербанк ****2231 и ВТБ ****7710."),
            ("assistant", "На какую карту?"),
            ("user", "На ту, куда приходит зарплата, — это не Сбер."),
        ], tags=("resolved_alternatives", "negation", "paired_details"),
            amount_rub=fill(15000, "m1"), destination_bank=fill("ВТБ", "m1", "m3"),
            destination_card_masked=fill("****7710", "m1", "m3"),
            source_account_kind=fill("расчётный счёт", "m1"), recipient_relation=fill("self", "m1")),

        scenario("H008", "transfer_draft", "Цитата чужой просьбы с другой суммой и картой не относится к переводу.", [
            ("user", "Переведи себе 6 400 рублей на мою карту ****8812 в Райффайзенбанке. В чате бухгалтер писал: "
             "«переведи 64 000 на карту ****1199» — это про другой платёж, его не трогай."),
        ], tags=("quoted_foreign_instruction", "ownership", "self_transfer"),
            amount_rub=fill(6400, "m1"), destination_bank=fill("Райффайзенбанк", "m1"),
            destination_card_masked=fill("****8812", "m1"), recipient_relation=fill("self", "m1")),

        scenario("H009", "invoice_draft", "Отсылка к прошлому счёту без названной суммы: сумму нельзя выдумать.", [
            ("user", "Выставь счёт ООО Мята за уборку офиса по той же цене, что в прошлом месяце."),
            ("assistant", "Какую сумму указать?"),
            ("user", "Посмотри сам в прошлом счёте, он у тебя есть."),
        ], tags=("reference_without_value", "missing_amount"),
            recipient_name=fill("ООО Мята", "m1"), subject=fill("уборка офиса", "m1"),
            items=fill([item("уборка офиса")], "m1")),

        scenario("H010", "none", "Пользователь откладывает счёт до согласования: запрос отменён.", [
            ("user", "Выставь счёт ООО Кедровый бор за 4 сеанса массажа по 3 500 рублей, без НДС."),
            ("assistant", "Итог 14 000 рублей. Выставляю?"),
            ("user", "Подожди, сначала согласую с бухгалтерией. Пока ничего не выставляй, я вернусь позже."),
        ], tags=("cancelled_request",), reason=NONE_REASON_TEXT["cancelled"]),
    ]


def scenarios():
    return manual_records() + generated()


# Семейства H011–H090: числа и эталон вычисляются из параметров, формулировки чередуются.
NAMES = ["ООО Северная звезда", "ООО Ромашка-Трейд", "ООО Каменный мост", "ООО Синий кит", "ООО Зелёный сад",
         "ООО Медный век", "ООО Полярный круг", "ООО Речной порт", "ООО Светлый дом", "ООО Тёплый край",
         "ООО Горный ручей", "ООО Лунная долина", "ООО Быстрый старт", "ООО Белая сова", "ООО Красный клён",
         "ООО Вершина", "ООО Орион-М", "ООО Дельта-Сервис", "ООО Пятый элемент", "ООО Новая линия",
         "ООО Морской бриз", "ООО Старый город", "ООО Ясный день", "ООО Золотое поле", "ООО Ветер перемен",
         "ООО Тихий берег", "ООО Кедр-Плюс", "ООО Сосновый бор", "ООО Летний сад", "ООО Чистый лист",
         "АО Восточный терминал", "АО Уральский цех", "АО Волжский берег", "ООО Северный ветер", "ООО Янтарь-М"]
GOODS = [("палатки", "палатка", 12400), ("спальники", "спальник", 3100), ("кресла", "кресло", 8700),
         ("стулья", "стул", 3200), ("столы", "стол", 9500), ("мониторы", "монитор", 15800),
         ("клавиатуры", "клавиатура", 2300), ("лампы", "лампа", 1450), ("коврики", "коврик", 640),
         ("папки", "папка", 180), ("короба", "короб", 310), ("картриджи", "картридж", 4200),
         ("стеллажи", "стеллаж", 7600), ("тумбы", "тумба", 5400), ("шкафы", "шкаф", 18900),
         ("чайники", "чайник", 2700), ("рюкзаки", "рюкзак", 3900), ("термосы", "термос", 1600),
         ("фонари", "фонарь", 1150), ("доски", "доска", 2900)]
PURPOSES = ["Оснащение офиса", "Снаряжение для похода", "Мебель для переговорной", "Обустройство склада",
            "Расходники для отдела", "Подготовка к выставке", "Оборудование для кухни", "Закупка для филиала"]
SERVICES = [("аудит отчётности", "аудит отчётности"), ("консультацию по налогам", "консультация по налогам"),
            ("разработку сайта", "разработка сайта"), ("обучение персонала", "обучение персонала"),
            ("уборку офиса", "уборка офиса"), ("настройку CRM", "настройка CRM"),
            ("перевод документов", "перевод документов"), ("ремонт кровли", "ремонт кровли"),
            ("доставку груза", "доставка груза"), ("юридическое сопровождение", "юридическое сопровождение")]
HOURLY = [("консультацию", "консультация"), ("аудит", "аудит"), ("монтаж", "монтаж"), ("настройку", "настройка"),
          ("обучение", "обучение")]
BANKS = [("Сбербанк", "в Сбербанке", "Сбер"), ("ВТБ", "в ВТБ", "ВТБ"), ("Альфа-Банк", "в Альфа-Банке", "Альфа"),
         ("Райффайзенбанк", "в Райффайзенбанке", "Райффайзен"), ("Газпромбанк", "в Газпромбанке", "Газпромбанк"),
         ("Т-Банк", "в Т-Банке", "Т-Банк"), ("Совкомбанк", "в Совкомбанке", "Совкомбанк")]
BIKS = ["044525225", "044030653", "044525593", "044525974", "046577674", "040349602"]


def money(value) -> str:
    """Как пишут суммы люди: разряды через пробел, дробная часть через запятую."""
    if isinstance(value, float) and not value.is_integer():
        return str(value).replace(".", ",")
    value = int(value)
    return f"{value:,}".replace(",", " ") if value >= 10000 else str(value)


def number(value):
    return int(value) if float(value).is_integer() else value


def card(index: int) -> str:
    return f"{(index * 7919 + 1234) % 9000 + 1000}"


def inn10(prefix9: str) -> str:
    weights = [2, 4, 10, 3, 5, 9, 4, 6, 8]
    return prefix9 + str(sum(w * int(d) for w, d in zip(weights, prefix9)) % 11 % 10)


def account(bik: str, index: int) -> str:
    base = f"40702810{{}}{index * 7307 % 10 ** 11:011}"
    return next(base.format(key) for key in range(10) if valid_account(base.format(key), bik))


def pick(pool, index, offset=0):
    # Отдельный хеш на параметр: умножение по модулю давало 5 покупателей из 35 и один банк из 7.
    digest = hashlib.sha256(f"{index}:{offset}".encode()).digest()
    return pool[int.from_bytes(digest[:4], "big") % len(pool)]


def two_goods(index, *, round_prices=False):
    pool = [good for good in GOODS if good[2] % 100 == 0] if round_prices else GOODS
    first = pick(pool, index)
    second = next(good for step in range(1, len(pool)) for good in [pick(pool, index, step * 3)] if good != first)
    return first, second


def same_price_change(case_id, split, index):
    (a_pl, a_sg, a_price), (b_pl, b_sg, b_price) = two_goods(index)
    name, purpose = pick(NAMES, index), pick(PURPOSES, index)
    q1, q2, qb = 2 + index % 5, 6 + index % 7, 3 + index % 4
    last = [f"{a_pl.capitalize()} — не {q1}, а {q2} шт., по той же цене.",
            f"Количество поменялось: {a_pl} нужно {q2} шт. вместо {q1}, цена прежняя.",
            f"Поправка по первой позиции: {q2} шт., а не {q1}. Цена как раньше."][index % 3]
    amount = q2 * a_price + qb * b_price
    return scenario(case_id, "invoice_draft", f"Отсылка к прежней цене после смены количества: {q2} × {a_price} + {qb} × {b_price} = {amount}.", [
        ("user", f"Выставь счёт {name}: {a_pl} — {q1} шт. по {money(a_price)} рублей. Назначение: {purpose}. Без НДС."),
        ("assistant", "Что-нибудь добавить?"),
        ("user", f"Добавь {b_pl} — {qb} шт. по {money(b_price)} рублей."),
        ("assistant", "Добавил."),
        ("user", last),
    ], split=split, tags=("reference_to_earlier_value", "quantity_correction", "arithmetic"),
        recipient_name=fill(name, "m1"), subject=fill(purpose, "m1"),
        items=fill([item(a_sg, q2, "шт.", a_price), item(b_sg, qb, "шт.", b_price)], "m1", "m3", "m5"),
        amount_rub=fill(amount, "m1", "m3", "m5", derivation=f"{q2} * {a_price} + {qb} * {b_price} = {amount} RUB"),
        vat_mode=fill("no_vat", "m1"))


def reverted_correction(case_id, split, index):
    (a_pl, a_sg, a_price), (b_pl, b_sg, b_price) = two_goods(index, round_prices=True)
    name, purpose = pick(NAMES, index, 2), pick(PURPOSES, index, 1)
    q1, q2, qb = 3 + index % 6, 10 + index % 5, 2 + index % 3
    b_new = b_price - 100 * (1 + index % 4)
    last = [f"Хотя нет, верни количество, которое я писал в самом начале. А {b_pl} по {money(b_new)} рублей — договорились о скидке.",
            f"Отбой по количеству — оставь как было в первом сообщении. И цена на {b_pl} теперь {money(b_new)} рублей."][index % 2]
    amount = q1 * a_price + qb * b_new
    return scenario(case_id, "invoice_draft", f"Поправка количества отменена отсылкой к первому сообщению; новая цена второй позиции: {q1} × {a_price} + {qb} × {b_new} = {amount}.", [
        ("user", f"Подготовь счёт {name}: {a_pl} — {q1} шт. по {money(a_price)} рублей, {b_pl} — {qb} шт. по {money(b_price)} рублей. "
         f"Назначение: {purpose}. Без НДС."),
        ("assistant", "Всё верно?"),
        ("user", f"{a_pl.capitalize()} сделай {q2}."),
        ("assistant", f"Изменил количество на {q2}."),
        ("user", last),
    ], split=split, tags=("reverted_correction", "reference_to_earlier_value", "price_correction", "arithmetic"),
        recipient_name=fill(name, "m1"), subject=fill(purpose, "m1"),
        items=fill([item(a_sg, q1, "шт.", a_price), item(b_sg, qb, "шт.", b_new)], "m1", "m5"),
        amount_rub=fill(amount, "m1", "m5", derivation=f"{q1} * {a_price} + {qb} * {b_new} = {amount} RUB"),
        vat_mode=fill("no_vat", "m1"))


def seller_decoy(case_id, split, index):
    seller, buyer = pick(NAMES, index, 4), pick(NAMES, index, 9)
    service_acc, service = pick(SERVICES, index)
    total = 15000 + 2500 * (index % 13)
    seller_bik, buyer_bik = pick(BIKS, index), pick(BIKS, index, 1)
    seller_inn = inn10(f"77{index % 1000:03}{index * 37 % 10000:04}")
    buyer_inn = inn10(f"78{index % 1000:03}{index * 53 % 10000:04}")
    seller_account, buyer_account = account(seller_bik, index), account(buyer_bik, index + 500)
    email = f"buh{index}@client{index}.example"
    seller_text = f"{seller}, ИНН {seller_inn}, р/с {seller_account}, БИК {seller_bik}"
    buyer_text = f"ИНН {buyer_inn}, счёт {buyer_account}, БИК {buyer_bik}, почта {email}"
    if index % 2:
        messages = [("user", f"Нужен счёт на оплату за {service_acc}, {money(total)} рублей, без НДС. Для шапки наши реквизиты: {seller_text}."),
                    ("assistant", "Кто покупатель?"),
                    ("user", f"Покупатель — {buyer}: {buyer_text}.")]
        sources = {"name": "m3", "service": "m1"}
    else:
        messages = [("user", f"Выставь счёт {buyer} за {service_acc}, {money(total)} рублей, без НДС."),
                    ("assistant", "Пришлите реквизиты."),
                    ("user", f"Сначала наши, для шапки: {seller_text}."),
                    ("assistant", "А реквизиты покупателя?"),
                    ("user", f"У покупателя {buyer_text}.")]
        sources = {"name": "m1", "service": "m1"}
    requisites = "m3" if index % 2 else "m5"
    return scenario(case_id, "invoice_draft", "Полные реквизиты продавца с верными контрольными разрядами не относятся к покупателю.",
                    messages, split=split, tags=("ownership", "seller_requisites_decoy", "complete_details"),
                    recipient_name=fill(buyer, sources["name"]), recipient_inn=fill(buyer_inn, requisites),
                    recipient_account=fill(buyer_account, requisites), recipient_bik=fill(buyer_bik, requisites),
                    recipient_email=fill(email, requisites), subject=fill(service, "m1"),
                    items=fill([item(service)], "m1"), amount_rub=fill(total, "m1"), vat_mode=fill("no_vat", "m1"))


def partial_discount(case_id, split, index):
    (a_pl, a_sg, a_price), (b_pl, b_sg, b_price) = two_goods(index + 3, round_prices=True)
    name, purpose = pick(NAMES, index, 6), pick(PURPOSES, index, 3)
    qa, qb, discount = 2 + index % 4, 3 + index % 5, (5, 10, 15, 20)[index % 4]
    amount = qa * a_price * (100 - discount) // 100 + qb * b_price
    return scenario(case_id, "invoice_draft", f"Скидка {discount}% только на одну позицию и только в итоге: {amount}.", [
        ("user", f"Счёт {name}: {a_pl} — {qa} шт. по {money(a_price)} рублей, {b_pl} — {qb} шт. по {money(b_price)} рублей. "
         f"Назначение: {purpose}. Без НДС."),
        ("assistant", "Есть скидки?"),
        ("user", f"На {a_pl} скидка {discount}% — отрази её в итоге, цены в строках не меняй. {b_pl.capitalize()} без скидки."),
    ], split=split, tags=("partial_discount", "arithmetic", "multiple_items"),
        recipient_name=fill(name, "m1"), subject=fill(purpose, "m1"),
        items=fill([item(a_sg, qa, "шт.", a_price), item(b_sg, qb, "шт.", b_price)], "m1", "m3"),
        amount_rub=fill(amount, "m1", "m3",
                        derivation=f"{qa} * {a_price} * (100 - {discount}) / 100 + {qb} * {b_price} = {amount} RUB"),
        vat_mode=fill("no_vat", "m1"))


def thousands_correction(case_id, split, index):
    bank, bank_in, _ = pick(BANKS, index)
    first, final = 20.5 + index % 9, (12.5 + index % 7, 18 + index % 5)[index % 2]
    final_number = number(final)
    return scenario(case_id, "transfer_draft", f"Сумма в тысячах с поправкой: {money(final)} тыс. = {int(final * 1000)}.", [
        ("user", f"С расчётного счёта переведи себе {money(first)} тыс. рублей на карту ****{card(index)} {bank_in}."),
        ("assistant", f"Подтверждаете {money(int(first * 1000))} рублей?"),
        ("user", f"Нет, {money(final_number)} тыс., а не {money(first)}."),
    ], split=split, tags=("thousands_shorthand", "latest_correction", "assistant_not_evidence"),
        amount_rub=fill(int(final * 1000), "m3", derivation=f"{final_number} * 1000 = {int(final * 1000)} RUB"),
        destination_bank=fill(bank, "m1"), destination_card_masked=fill(f"****{card(index)}", "m1"),
        source_account_kind=fill("расчётный счёт", "m1"), recipient_relation=fill("self", "m1"))


def withdrawn_added(case_id, split, index):
    (a_pl, a_sg, a_price), (b_pl, b_sg, b_price) = two_goods(index + 5)
    c_pl, c_sg, c_price = next(good for step in range(2, len(GOODS)) for good in [pick(GOODS, index + 5, step * 5)]
                               if good[1] not in (a_sg, b_sg))
    name, purpose = pick(NAMES, index, 11), pick(PURPOSES, index, 5)
    qa, qb, qc = 4 + index % 6, 5 + index % 3, 2 + index % 4
    amount = qa * a_price + qc * c_price
    return scenario(case_id, "invoice_draft", f"Позиция снята, другая добавлена позже: {qa} × {a_price} + {qc} × {c_price} = {amount}.", [
        ("user", f"Счёт {name}: {a_pl} — {qa} шт. по {money(a_price)} рублей, {b_pl} — {qb} шт. по {money(b_price)} рублей. "
         f"Назначение: {purpose}. Без НДС."),
        ("assistant", "Всё в одном счёте?"),
        ("user", f"{b_pl.capitalize()} убери, их купим отдельно."),
        ("assistant", "Убрал."),
        ("user", f"Зато добавь {c_pl} — {qc} шт. по {money(c_price)} рублей."),
    ], split=split, tags=("withdrawn_item", "added_item", "arithmetic"),
        recipient_name=fill(name, "m1"), subject=fill(purpose, "m1"),
        items=fill([item(a_sg, qa, "шт.", a_price), item(c_sg, qc, "шт.", c_price)], "m1", "m3", "m5"),
        amount_rub=fill(amount, "m1", "m3", "m5", derivation=f"{qa} * {a_price} + {qc} * {c_price} = {amount} RUB"),
        vat_mode=fill("no_vat", "m1"))


def negation_choice(case_id, split, index):
    (bank_a, _, short_a), (bank_b, _, _) = pick(BANKS, index), pick(BANKS, index, 3)
    card_a, card_b = card(index), card(index + 40)
    total = 5000 + 1500 * (index % 11)
    last = [f"На ту, куда приходит зарплата, — это не {short_a}.", f"Не в {short_a}, на вторую."][index % 2]
    return scenario(case_id, "transfer_draft", f"Карта выбрана через отрицание: {bank_b} ****{card_b}.", [
        ("user", f"Переведи себе {money(total)} рублей с расчётного счёта. Карты у меня две: {bank_a} ****{card_a} и {bank_b} ****{card_b}."),
        ("assistant", "На какую карту?"),
        ("user", last),
    ], split=split, tags=("resolved_alternatives", "negation", "paired_details"),
        amount_rub=fill(total, "m1"), destination_bank=fill(bank_b, "m1", "m3"),
        destination_card_masked=fill(f"****{card_b}", "m1", "m3"),
        source_account_kind=fill("расчётный счёт", "m1"), recipient_relation=fill("self", "m1"))


def quoted_foreign(case_id, split, index):
    bank, bank_in, _ = pick(BANKS, index, 2)
    total, other = 3000 + 400 * (index % 17), 30000 + 4000 * (index % 9)
    who = ["бухгалтер", "партнёр", "коллега", "сестра"][index % 4]
    return scenario(case_id, "transfer_draft", "Цитата чужой просьбы с другой суммой и картой не относится к переводу.", [
        ("user", f"Переведи себе {money(total)} рублей на мою карту ****{card(index)} {bank_in}. В чате {who} писал: "
         f"«переведи {money(other)} на карту ****{card(index + 70)}» — это про другой платёж, его не трогай."),
    ], split=split, tags=("quoted_foreign_instruction", "ownership", "self_transfer"),
        amount_rub=fill(total, "m1"), destination_bank=fill(bank, "m1"),
        destination_card_masked=fill(f"****{card(index)}", "m1"), recipient_relation=fill("self", "m1"))


def reference_without_value(case_id, split, index):
    name = pick(NAMES, index, 13)
    service_acc, service = pick(SERVICES, index, 4)
    when = ["в прошлом месяце", "в прошлый раз", "в мартовском счёте"][index % 3]
    last = ["Посмотри сам в прошлом счёте, он у тебя есть.", "Как обычно, ты же знаешь сумму."][index % 2]
    return scenario(case_id, "invoice_draft", "Отсылка к прошлой цене без названной суммы: сумму нельзя выдумать.", [
        ("user", f"Выставь счёт {name} за {service_acc} по той же цене, что {when}."),
        ("assistant", "Какую сумму указать?"),
        ("user", last),
    ], split=split, tags=("reference_without_value", "missing_amount"),
        recipient_name=fill(name, "m1"), subject=fill(service, "m1"), items=fill([item(service)], "m1"))


def cancelled(case_id, split, index):
    name = pick(NAMES, index, 15)
    service_acc, _ = pick(SERVICES, index, 6)
    total = 8000 + 1000 * (index % 9)
    last = ["Подожди, сначала согласую с бухгалтерией. Пока ничего не выставляй.",
            "Стоп, клиент отказался. Счёт не нужен.",
            "Отмени, пожалуйста, они уже оплатили по старому счёту."][index % 3]
    return scenario(case_id, "none", "Пользователь отменяет запрос на счёт.", [
        ("user", f"Выставь счёт {name} за {service_acc} на {money(total)} рублей, без НДС."),
        ("assistant", f"Итог {money(total)} рублей. Выставляю?"),
        ("user", last),
    ], split=split, tags=("cancelled_request",), reason=NONE_REASON_TEXT["cancelled"])


def two_corrections(case_id, split, index):
    (a_pl, a_sg, a_price), (b_pl, b_sg, b_price) = two_goods(index + 7)
    name, purpose = pick(NAMES, index, 17), pick(PURPOSES, index, 6)
    q1, q2, qb = 2 + index % 4, 7 + index % 6, 1 + index % 5
    new_price = a_price + 100 * (2 + index % 5)
    amount = q2 * new_price + qb * b_price
    return scenario(case_id, "invoice_draft", f"Количество и цена первой позиции исправлены в разных репликах: {q2} × {new_price} + {qb} × {b_price} = {amount}.", [
        ("user", f"Подготовь счёт {name}: {a_pl} — {q1} шт. по {money(a_price)} рублей, {b_pl} — {qb} шт. по {money(b_price)} рублей. "
         f"Назначение: {purpose}. Без НДС."),
        ("assistant", "Записал."),
        ("user", f"{a_pl.capitalize()} — {q2} шт."),
        ("assistant", "Изменил количество."),
        ("user", f"И цена на {a_pl} выросла до {money(new_price)} рублей, остальное без изменений."),
    ], split=split, tags=("multiple_corrections", "price_correction", "arithmetic"),
        recipient_name=fill(name, "m1"), subject=fill(purpose, "m1"),
        items=fill([item(a_sg, q2, "шт.", new_price), item(b_sg, qb, "шт.", b_price)], "m1", "m3", "m5"),
        amount_rub=fill(amount, "m1", "m3", "m5", derivation=f"{q2} * {new_price} + {qb} * {b_price} = {amount} RUB"),
        vat_mode=fill("no_vat", "m1"))


def hourly_correction(case_id, split, index):
    name = pick(NAMES, index, 19)
    service_acc, service = pick(HOURLY, index)
    rate, hours, final = 1800 + 200 * (index % 8), 3 + index % 3, (4.5, 2.5, 6.5, 5.5)[index % 4]
    amount = int(final * rate)
    return scenario(case_id, "invoice_draft", f"Дробные часы после поправки: {money(final)} × {rate} = {amount}.", [
        ("user", f"Счёт {name} за {service_acc}: {hours} ч. по {money(rate)} рублей за час. Без НДС."),
        ("assistant", "Проверьте часы."),
        ("user", f"Поправка: не {hours}, а {money(final)} ч."),
    ], split=split, tags=("decimal_quantity", "latest_correction", "arithmetic"),
        recipient_name=fill(name, "m1"), subject=fill(service, "m1"),
        items=fill([item(service, final, "ч.", rate)], "m1", "m3"),
        amount_rub=fill(amount, "m1", "m3", derivation=f"{final} * {rate} = {amount} RUB"),
        vat_mode=fill("no_vat", "m1"))


def summed_transfer(case_id, split, index):
    bank, bank_in, _ = pick(BANKS, index, 5)
    first, extra = 9000 + 1000 * (index % 8), 1500 + 500 * (index % 6)
    return scenario(case_id, "transfer_draft", f"Сумма собирается из двух реплик: {first} + {extra} = {first + extra}.", [
        ("user", f"Переведи себе {money(first)} рублей с расчётного счёта на карту ****{card(index)} {bank_in}."),
        ("assistant", "Готово к подтверждению."),
        ("user", f"Добавь к этой сумме ещё {money(extra)} рублей — одним переводом."),
    ], split=split, tags=("sum_across_turns", "arithmetic"),
        amount_rub=fill(first + extra, "m1", "m3", derivation=f"{first} + {extra} = {first + extra} RUB"),
        destination_bank=fill(bank, "m1"), destination_card_masked=fill(f"****{card(index)}", "m1"),
        source_account_kind=fill("расчётный счёт", "m1"), recipient_relation=fill("self", "m1"))


def control_invoice(case_id, split, index):
    (a_pl, a_sg, a_price), (b_pl, b_sg, b_price) = two_goods(index + 11)
    name, purpose = pick(NAMES, index, 21), pick(PURPOSES, index, 7)
    qa, qb = 2 + index % 5, 1 + index % 3
    amount = qa * a_price + qb * b_price
    return scenario(case_id, "invoice_draft", f"Контроль ложных замечаний: всё в одной реплике, {amount}.", [
        ("user", f"Выставь счёт {name}: {a_pl} — {qa} шт. по {money(a_price)} рублей, {b_pl} — {qb} шт. по {money(b_price)} рублей. "
         f"Назначение: {purpose}. Без НДС."),
    ], split=split, tags=("control", "arithmetic"),
        recipient_name=fill(name, "m1"), subject=fill(purpose, "m1"),
        items=fill([item(a_sg, qa, "шт.", a_price), item(b_sg, qb, "шт.", b_price)], "m1"),
        amount_rub=fill(amount, "m1", derivation=f"{qa} * {a_price} + {qb} * {b_price} = {amount} RUB"),
        vat_mode=fill("no_vat", "m1"))


def control_transfer(case_id, split, index):
    bank, bank_in, _ = pick(BANKS, index, 6)
    total = 4000 + 700 * (index % 10)
    return scenario(case_id, "transfer_draft", "Контроль ложных замечаний: простой перевод себе.", [
        ("user", f"С расчётного счёта переведи себе {money(total)} рублей на карту ****{card(index)} {bank_in}."),
    ], split=split, tags=("control", "self_transfer"),
        amount_rub=fill(total, "m1"), destination_bank=fill(bank, "m1"),
        destination_card_masked=fill(f"****{card(index)}", "m1"),
        source_account_kind=fill("расчётный счёт", "m1"), recipient_relation=fill("self", "m1"))


def two_distinct(low, high, index, offset):
    """Два разных числа из диапазона: для поправок и ошибочных пересказов."""
    first = low + int(hashlib.sha256(f"{index}:{offset}".encode()).hexdigest()[:6], 16) % (high - low)
    second = low + (first - low + 1 + index % (high - low - 1)) % (high - low)
    return first, second


def long_scattered(case_id, split, index):
    (a_pl, a_sg, a_price), (b_pl, b_sg, b_price) = two_goods(index + 13)
    name, purpose, bik = pick(NAMES, index, 23), pick(PURPOSES, index, 9), pick(BIKS, index, 4)
    qa, qb = 2 + index % 7, 3 + index % 5
    inn, acc = inn10(f"79{index % 1000:03}{index * 41 % 10000:04}"), account(bik, index + 900)
    email = f"zakaz{index}@buyer{index}.example"
    amount = (qa + qb) * a_price
    return scenario(case_id, "invoice_draft", f"Данные разбросаны по 13 репликам, вторая позиция «по той же цене, что первая»: {qa} × {a_price} + {qb} × {a_price} = {amount}.", [
        ("user", f"Здравствуйте! Нужен счёт для {name}."),
        ("assistant", "Конечно. Что включить в счёт?"),
        ("user", "Сейчас уточню у коллег, минуту."),
        ("assistant", "Хорошо, жду."),
        ("user", f"Так, первая позиция: {a_pl} — {qa} шт. по {money(a_price)} рублей."),
        ("assistant", "Записал. Что ещё?"),
        ("user", "Кстати, а когда обычно приходят деньги по таким счетам?"),
        ("assistant", "Обычно в течение одного-трёх рабочих дней."),
        ("user", f"Понятно. Вторая позиция: {b_pl} — {qb} шт., по той же цене, что первая. Назначение: {purpose}."),
        ("assistant", "Добавил. Счёт с НДС?"),
        ("user", f"Без НДС. Реквизиты покупателя: ИНН {inn}, счёт {acc}, БИК {bik}."),
        ("assistant", "Спасибо. Куда отправить счёт?"),
        ("user", f"На почту {email}."),
    ], split=split, tags=("long_scattered", "long_dialogue", "distractor_questions", "reference_to_earlier_value", "arithmetic"),
        recipient_name=fill(name, "m1"), recipient_inn=fill(inn, "m11"), recipient_account=fill(acc, "m11"),
        recipient_bik=fill(bik, "m11"), recipient_email=fill(email, "m13"), subject=fill(purpose, "m9"),
        items=fill([item(a_sg, qa, "шт.", a_price), item(b_sg, qb, "шт.", a_price)], "m5", "m9"),
        amount_rub=fill(amount, "m5", "m9", derivation=f"{qa} * {a_price} + {qb} * {a_price} = {amount} RUB"),
        vat_mode=fill("no_vat", "m11"))


def long_corrections(case_id, split, index):
    (a_pl, a_sg, a_price), (b_pl, b_sg, b_price) = two_goods(index + 17, round_prices=True)
    name, purpose = pick(NAMES, index, 25), pick(PURPOSES, index, 10)
    q1, q2 = two_distinct(2, 9, index, 1)
    q3 = 10 + index % 6
    qb = 2 + index % 4
    discounted = b_price - 100 * (1 + index % 5)
    discount = (5, 10, 15, 20)[index % 4]
    amount = q3 * a_price * (100 - discount) // 100 + qb * b_price
    return scenario(case_id, "invoice_draft", f"Поправки в 13 репликах, отмена цены и скидка {discount}% на одну позицию в итоге: {amount}.", [
        ("user", f"Подготовь счёт {name}: {a_pl} — {q1} шт. по {money(a_price)} рублей, {b_pl} — {qb} шт. по {money(b_price)} рублей. "
         f"Назначение: {purpose}. Без НДС."),
        ("assistant", "Записал."),
        ("user", f"{a_pl.capitalize()} — {q2} шт."),
        ("assistant", "Изменил."),
        ("user", "Подожди, мне звонят."),
        ("assistant", "Жду."),
        ("user", f"Так, цена на {b_pl} — {money(discounted)} рублей, договорились."),
        ("assistant", "Обновил цену."),
        ("user", f"И {a_pl} всё-таки {q3} шт."),
        ("assistant", "Изменил количество."),
        ("user", f"Хотя нет, {b_pl}: верни цену из первого сообщения, скидку не одобрили."),
        ("assistant", "Вернул прежнюю цену."),
        ("user", f"Всё. И на {a_pl} скидка {discount}% — только в итоге, цены в строках не меняй. Теперь можно выставлять."),
    ], split=split, tags=("long_multiple_corrections", "long_dialogue", "reverted_correction", "partial_discount", "arithmetic"),
        recipient_name=fill(name, "m1"), subject=fill(purpose, "m1"),
        items=fill([item(a_sg, q3, "шт.", a_price), item(b_sg, qb, "шт.", b_price)], "m1", "m9", "m11", "m13"),
        amount_rub=fill(amount, "m1", "m9", "m11", "m13",
                        derivation=f"{q3} * {a_price} * (100 - {discount}) / 100 + {qb} * {b_price} = {amount} RUB"),
        vat_mode=fill("no_vat", "m1"))


def wrong_summary(case_id, split, index):
    (a_pl, a_sg, a_price), (b_pl, b_sg, b_price) = two_goods(index + 19, round_prices=True)
    name, purpose = pick(NAMES, index, 27), pick(PURPOSES, index, 11)
    qa, qa_wrong = two_distinct(2, 9, index, 2)
    qb = 2 + index % 5
    wrong_price = b_price + 100 * (1 + index % 4)
    c_pl, c_sg, _ = next(good for step in range(2, len(GOODS)) for good in [pick(GOODS, index, 30 + step)]
                         if good[1] not in (a_sg, b_sg))
    qc = 1 + index % 3
    amount = qa * a_price + (qb + qc) * b_price
    return scenario(case_id, "invoice_draft", f"Ассистент пересказал с ошибкой; новая позиция по исправленной цене, а не по цене ассистента: {amount}.", [
        ("user", f"Счёт {name}: {a_pl} — {qa} шт. по {money(a_price)} рублей, {b_pl} — {qb} шт. по {money(b_price)} рублей. "
         f"Назначение: {purpose}. Без НДС."),
        ("assistant", f"Итого: {a_pl} — {qa_wrong} шт., {b_pl} — {qb} шт. по {money(wrong_price)} рублей. Всё верно?"),
        ("user", "Почти. Подожди, сверю с накладной."),
        ("assistant", "Хорошо."),
        ("user", "Сначала скажи: назначение потом можно будет поменять?"),
        ("assistant", "Да, до отправки счёта можно."),
        ("user", f"Хорошо. {a_pl.capitalize()} — {qa} шт., как я писал."),
        ("assistant", "Исправил количество."),
        ("user", f"И цена на {b_pl} {money(b_price)} рублей, а не {money(wrong_price)}. "
         f"Добавь ещё {c_pl} — {qc} шт. по той же цене, что {b_pl}."),
    ], split=split, tags=("assistant_wrong_summary", "long_dialogue", "assistant_not_evidence", "reference_to_earlier_value", "arithmetic"),
        recipient_name=fill(name, "m1"), subject=fill(purpose, "m1"),
        items=fill([item(a_sg, qa, "шт.", a_price), item(b_sg, qb, "шт.", b_price), item(c_sg, qc, "шт.", b_price)],
                   "m1", "m7", "m9"),
        amount_rub=fill(amount, "m1", "m7", "m9",
                        derivation=f"{qa} * {a_price} + {qb} * {b_price} + {qc} * {b_price} = {amount} RUB"),
        vat_mode=fill("no_vat", "m1"))


def topic_switch(case_id, split, index):
    (a_pl, a_sg, a_price), (b_pl, b_sg, b_price) = two_goods(index + 23)
    name, purpose = pick(NAMES, index, 29), pick(PURPOSES, index, 12)
    qa, qb = 2 + index % 6, 1 + index % 4
    other = [15000 + 1000 * (index % 9), 4000 + 500 * (index % 7), 20000 + 2500 * (index % 5)]
    delivery = 500 + 250 * (index % 6)
    amount = qa * a_price + qb * b_price + delivery
    return scenario(case_id, "invoice_draft", f"Посреди счёта обсуждаются другие платежи; доставка добавлена к итогу без отдельной строки: {amount}.", [
        ("user", f"Выставь счёт {name}: {a_pl} — {qa} шт. по {money(a_price)} рублей. Назначение: {purpose}. Без НДС."),
        ("assistant", "Записал. Что-то ещё?"),
        ("user", f"Пока отвлекусь: вчера я переводил поставщику {money(other[0])} рублей, а сегодня ещё {money(other[1])} — это всё прошло?"),
        ("assistant", "Детали прошлых платежей видны в истории операций."),
        ("user", f"Ладно. А если оплатить {money(other[2])} рублей до пятницы, будет скидка?"),
        ("assistant", "Условия скидок уточните у менеджера."),
        ("user", f"Хорошо. Так, по счёту: добавь {b_pl} — {qb} шт. по {money(b_price)} рублей."),
        ("assistant", "Добавил."),
        ("user", f"И ещё раз про перевод: {money(other[0])} ушли на карту поставщика, да?"),
        ("assistant", "Проверьте историю операций."),
        ("user", f"Ок. По счёту: прибавь к итоговой сумме доставку {money(delivery)} рублей, отдельной позицией её не выделяй. "
         "Теперь выставляй."),
    ], split=split, tags=("topic_switch", "long_dialogue", "distractor_numbers", "extra_charge", "arithmetic"),
        recipient_name=fill(name, "m1"), subject=fill(purpose, "m1"),
        items=fill([item(a_sg, qa, "шт.", a_price), item(b_sg, qb, "шт.", b_price)], "m1", "m7"),
        amount_rub=fill(amount, "m1", "m7", "m11",
                        derivation=f"{qa} * {a_price} + {qb} * {b_price} + {delivery} = {amount} RUB"),
        vat_mode=fill("no_vat", "m1"))


# Длинные диалоги добавлены после H090, чтобы не перенумеровать прежние кейсы: dev H091–H094, test H095–H104.
LONG_PLAN = [(long_scattered, 1, 3), (long_corrections, 1, 3), (wrong_summary, 1, 2), (topic_switch, 1, 2)]

# Сколько кейсов каждого семейства в dev (H011–H030) и test (H031–H090).
PLAN = [(same_price_change, 2, 6), (reverted_correction, 2, 6), (seller_decoy, 1, 5), (partial_discount, 2, 5),
        (thousands_correction, 1, 4), (withdrawn_added, 2, 5), (negation_choice, 1, 4), (quoted_foreign, 1, 4),
        (reference_without_value, 1, 4), (cancelled, 1, 4), (two_corrections, 2, 5), (hourly_correction, 1, 3),
        (summed_transfer, 1, 3), (control_invoice, 1, 2), (control_transfer, 1, 0)]


def generated():
    records, number_, openings = [], 11, {case["messages"][0]["text"] for case, _, _ in manual_records()}
    for plan, split, column in ((PLAN, "dev", 1), (PLAN, "test", 2), (LONG_PLAN, "dev", 1), (LONG_PLAN, "test", 2)):
        for family, *counts in plan:
            for _ in range(counts[column - 1]):
                case_id = f"H{number_:03}"
                # Индекс из хеша ID; при совпадении первой реплики с любым кейсом берётся следующий.
                index = int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16) % 100000
                record = family(case_id, split, index)
                while record[0]["messages"][0]["text"] in openings:
                    index += 1
                    record = family(case_id, split, index)
                openings.add(record[0]["messages"][0]["text"])
                records.append(record)
                number_ += 1
    return records


def valid_inn(value: str) -> bool:
    digits = [int(char) for char in value]

    def control(weights):
        return sum(weight * digit for weight, digit in zip(weights, digits)) % 11 % 10

    if len(digits) == 10:
        return control([2, 4, 10, 3, 5, 9, 4, 6, 8]) == digits[9]
    return (len(digits) == 12 and control([7, 2, 4, 10, 3, 5, 9, 4, 6, 8]) == digits[10]
            and control([3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]) == digits[11])


def valid_account(account: str, bik: str) -> bool:
    """Ключ расчётного счёта по последним трём цифрам БИК (кредитная организация)."""
    weights = [7, 1, 3] * 8
    return sum(int(char) * weights[index] for index, char in enumerate(bik[-3:] + account)) % 10 == 0


def build_contents():
    records = scenarios()
    cases, labels, notes = zip(*records)
    if [case["case_id"] for case in cases] != [f"H{i:03}" for i in range(1, len(cases) + 1)]:
        raise ValueError("Нужны уникальные последовательные ID H001…")
    for case, label, meta in records:
        parsed = InputCase.model_validate(case)
        validate_messages(parsed.messages)
        LABEL_ADAPTER.validate_python(label)
        user_ids = {m.id for m in parsed.messages if m.role == "user"}
        fields = label["fields"]
        for name, value in fields.items():
            if not set(value["evidence"]) <= user_ids:
                raise ValueError("Источник не является сообщением пользователя: " + parsed.case_id)
            if value["status"] == "missing" and value["evidence"] or value["status"] != "missing" and not value["evidence"]:
                raise ValueError("Источники не соответствуют статусу: " + parsed.case_id)
            if value["status"] == "fillable" and invalid_format(name, value["value"]):
                raise ValueError("Некорректный формат эталона: " + parsed.case_id + "." + name)
        inn = fields.get("recipient_inn", {}).get("value")
        if inn is not None and not valid_inn(inn):
            raise ValueError("Неверный контрольный разряд ИНН: " + parsed.case_id)
        account, bik = (fields.get(name, {}).get("value") for name in ("recipient_account", "recipient_bik"))
        if account is not None and (bik is None or not valid_account(account, bik)):
            raise ValueError("Неверный ключ счёта по БИК: " + parsed.case_id)
    openings = [case["messages"][0]["text"] for case in cases]
    if len(set(openings)) != len(openings):
        raise ValueError("Первые реплики кейсов повторяются")
    encode = lambda rows: "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    inputs, targets = encode(cases), encode(labels)
    manifest = {
        "dataset_version": 1, "name": "challenge", "synthetic": True,
        "authorship": "assistant_authored_before_model_runs", "human_reviewed": True,
        "annotation_status": "spot_checked_by_human",
        "human_review": "2026-11-04, до запуска на test: выборочная проверка 36 test-кейсов (по 2 на семейство) "
                        "по .scratch/challenge_review.md; замечаний нет. Остальные эталоны вычислены генератором.",
        "case_count": len(cases),
        "splits": {split: sum(meta["split"] == split for meta in notes) for split in ("dev", "test")},
        "evidence_rule": "all user messages that state or select the current value",
        "requisites": "fictional, with valid INN check digits and account keys for the stated BIK",
        "input_sha256": hashlib.sha256(inputs.encode()).hexdigest(),
        "labels_sha256": hashlib.sha256(targets.encode()).hexdigest(), "cases": list(notes),
    }
    catalogue = [
        "# Challenge: сложные диалоги для проверки извлечения и Jev", "",
        "Синтетические диалоги, построенные по реальным промахам: финал на evaluation50 показал 97% у llm-only,",
        "поэтому здесь собраны ситуации, где нужно связать несколько реплик или отличить чужие данные.",
        "Кейсы и эталоны составлены ассистентом до прогонов; 2026-11-04 человек выборочно проверил 36 test-кейсов (по 2 на семейство).",
        "Реквизиты вымышлены, но ИНН и ключ счёта по БИК корректны; почта использует домен .example.",
        "H001–H030 и H091–H094 — split=dev: для пилотов и настройки промптов; в итоговое сравнение не входят.",
        "H031–H090 и H095–H104 — split=test: не запускать до предрегистрации финального сравнения.",
        "H091–H104 — длинные диалоги (9–13 реплик) со встроенными ловушками: данные вразброс с отсылкой к цене,",
        "много поправок со скидкой, ошибочный пересказ ассистента, смена темы с доставкой в итоге.",
        "H001–H010 написаны вручную; H011–H090 построены по семействам сценариев с вычисленным эталоном.",
        "Ограничение: dev и test используют одни и те же семейства, поэтому test проверяет обобщение",
        "на новые параметры и формулировки внутри семейства, но не на новые типы ловушек.", "",
        "Правило источников: evidence перечисляет все реплики пользователя, где актуальное значение названо или выбрано.", "",
        "| ID | Split | Тип | Реплик | Проверяемое условие и эталонное решение |",
        "| --- | --- | --- | ---: | --- |",
    ]
    catalogue += [f"| {case['case_id']} | {meta['split']} | {label['document_type']} | {len(case['messages'])} | {meta['rationale']} |"
                  for case, label, meta in records]
    catalogue += ["", "Сборка: `uv run --locked python -m scripts.build_challenge_dataset`.",
                  "Проверка: `uv run --locked python -m scripts.build_challenge_dataset --check`.", ""]
    return {"cases_challenge.jsonl": inputs, "labels_challenge.jsonl": targets,
            "challenge.manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            "challenge.md": "\n".join(catalogue)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, contents in build_contents().items():
        path = DATA / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != contents:
                raise SystemExit(f"{name} не соответствует scripts.build_challenge_dataset")
        else:
            path.write_text(contents, encoding="utf-8")
    print("challenge: сценарии построены; требуется проверка человеком; API не вызывались.")


if __name__ == "__main__":
    main()
