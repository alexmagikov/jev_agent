"""Собирает фиксированную стресс-выборку из 30 существующих кейсов v2.

Запуск: python -m scripts.build_hard30_dataset [--check]. Без API и .env.
Сложность выбрана по содержанию сценариев, не по автоматическому рейтингу моделей.
Исходные строки, ID, split и разметка сохраняются без изменений.
"""

import argparse
import json
from pathlib import Path

from src.extraction import validate_messages
from src.main import load_cases, load_labels


DATA = Path(__file__).resolve().parents[1] / "data"

# Фиксированный манифест: причины выбора не передаются моделям.
SELECTION = {
    "C004": "Исправление суммы: актуальное значение вместо раннего.",
    "C026": "Цена единицы из суммы и количества; товар в кавычках.",
    "C027": "Несколько товаров, десятичные количества и цены в разных форматах.",
    "C042": "Одновременно реквизиты продавца и покупателя; разделение ролей.",
    "C043": "Последовательность нескольких явных исправлений одной суммы.",
    "C044": "Реквизиты и НДС выдуманы ассистентом, а не сообщены пользователем.",
    "C046": "Некорректные длины трёх реквизитов; invalid без дополнения цифр.",
    "C047": "Явные 0% НДС не равны режиму без НДС.",
    "C049": "Обратный расчёт цены единицы при неизвестной единице измерения.",
    "C051": "Двухступенчатый расчёт количества и общей суммы.",
    "C052": "Удаление позиции и изменение цены оставшейся; пересчёт суммы.",
    "C053": "Исправление даты без изменения номера с нулями, буквами и слешем.",
    "C055": "Отмена и новый запрос: реквизиты старого получателя не переносятся.",
    "C057": "Неразрешённый выбор получателя при известной сумме.",
    "C059": "Неоднозначная роль числа: цена единицы или всей партии.",
    "C060": "Общие количество и сумма не определяют атрибуты отдельных товаров.",
    "C063": "Вложенные кавычки, название, единица комплектации и расчёт суммы.",
    "C064": "Исторический чужой счёт, исправление количества и сохранение НДС.",
    "C065": "Цитируемая инструкция не является поручением и не меняет сумму/тип.",
    "C066": "Исправление суммы перевода на фоне упоминания старой операции.",
    "C067": "Карту и банк сообщил только ассистент; необоснованные заполнения.",
    "C068": "Две собственные карты в разных банках без выбора.",
    "C070": "Рубли и копейки: сохранение дробной суммы перевода.",
    "C072": "Отзыв карты без отмены перевода; остальные поля сохраняются.",
    "C073": "Замена счёта переводом себе: смена типа документа и суммы.",
    "C074": "Сложная маска карты: не восстанавливать скрытые цифры.",
    "C075": "Поздний выбор карты разрешает прежнюю неоднозначность банка и карты.",
    "C076": "Запрос инструкции, а не документа; инициатива ассистента не поручение.",
    "C078": "Отмена счёта с переходом к информационному вопросу.",
    "C081": "Замена счёта неподдерживаемым УПД: итоговый тип none.",
}


def build_contents():
    if len(SELECTION) != 30:
        raise ValueError("В стресс-выборке должно быть ровно 30 уникальных кейсов")
    selected = set(SELECTION)
    cases = load_cases(DATA / "cases_v2.jsonl", "all")
    missing = selected - {case.case_id for case in cases}
    if missing:
        raise ValueError("В cases_v2 нет выбранных кейсов: " + ", ".join(sorted(missing)))
    labels = load_labels(DATA / "labels_v2.jsonl", selected)
    labels_by_id = {label["case_id"]: label for label in labels}
    for case in cases:
        if case.case_id not in selected:
            continue
        validate_messages(case.messages)
        user_ids = {message.id for message in case.messages if message.role == "user"}
        for field in labels_by_id[case.case_id]["fields"].values():
            if not set(field["evidence"]) <= user_ids:
                raise ValueError("Источник разметки должен быть репликой пользователя: " + case.case_id)
    contents = {}
    for source, target in (("cases_v2.jsonl", "cases_hard30.jsonl"),
                           ("labels_v2.jsonl", "labels_hard30.jsonl")):
        original = (DATA / source).read_text(encoding="utf-8")
        contents[target] = "".join(
            line if line.endswith("\n") else line + "\n"
            for line in original.splitlines(keepends=True)
            if line.strip() and json.loads(line)["case_id"] in selected
        )
    return contents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Проверить файлы без перезаписи")
    args = parser.parse_args()
    for name, contents in build_contents().items():
        path = DATA / name
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != contents:
                raise SystemExit(f"{name} не соответствует выборке: запустите scripts.build_hard30_dataset")
        else:
            path.write_text(contents, encoding="utf-8")
    print("hard30: 30 кейсов, dev=17/test=13; 19 счетов, 8 переводов, 3 none. API не вызывались.")


if __name__ == "__main__":
    main()
