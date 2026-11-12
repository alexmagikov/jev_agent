"""Внешний ru-invoice-extraction-benchmark в формате пайплайна: текст счёта → одна реплика.

Источник: necrasov-ilya/ru-invoice-extraction-benchmark (Apache-2.0), копия ревизии
SOURCE_REVISION лежит в data/external. Эталон берётся только из разметки бенчмарка.
Поля, которых в бенчмарке нет, заполняются заглушкой missing: они нужны схеме
эталона, но не оцениваются (см. scripts/score_ru_invoice.py).

Запуск: python -m scripts.build_ru_invoice_dataset [--check]. API и .env не используются.
"""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.build_dataset_v2 import example, fill, item, unfilled
from src.extraction import invalid_format, validate_messages
from src.main import InputCase, LABEL_ADAPTER


DATA = Path(__file__).resolve().parents[1] / "data"
SOURCE = DATA / "external" / "ru-invoice-extraction-benchmark"
SOURCE_URL = "https://huggingface.co/datasets/necrasov-ilya/ru-invoice-extraction-benchmark"
SOURCE_REVISION = "097b4b1789ff6b90d1077cc5f67bff28e79b5fad"
SOURCE_SHA256 = {
    "data/development.jsonl": "586b4929be151fbc790b630e6c84cf0f7f4b09c8523ac9d6ebb5f7ef16b0d588",
    "data/validation.jsonl": "aae3b597e562a65d581084b9fa91ca48b886e8e1604dec4301f408e485ff0f8f",
    "data/test.jsonl": "7a5554a4e567efdffdcff6ad4c158cdf6a9c22d2d587d215cccbd58096f41f7f",
    "schema/invoice-v1.schema.json": "7b0d13a360e1a2823aeb030feba9768e03fb7d72c84d22cbfff300d82e731fba",
    "LICENSE": "50e6751797c50dedd75ef1b8a0d9e42f5f8472e9fbce91f34718e9f97b0c780a",
}
# Пользователь банка — поставщик: пересылает готовый документ и просит черновик счёта покупателю.
INPUT_TEMPLATE = "Подготовь черновик счёта по этому документу:\n\n{text}"
SPLITS = {"development": "dev", "validation": "dev", "test": "test"}
EXTERNAL_FIELDS = ("recipient_name", "recipient_inn", "items", "amount_rub", "vat_mode", "vat_rate_percent")
PLACEHOLDER_FIELDS = ("recipient_email", "recipient_account", "recipient_bik", "subject", "basis_number", "basis_date")
# Режим НДС бенчмарка → (vat_mode, vat_rate_percent) по разделу 5.5 контракта; смешанные ставки — одна ставка неоднозначна.
VAT = {
    "standard_20": (fill("rate", "m1"), fill(20, "m1")),
    "reduced_10": (fill("rate", "m1"), fill(10, "m1")),
    "zero_rate": (fill("rate", "m1"), fill(0, "m1")),
    "exempt": (fill("no_vat", "m1"), unfilled()),
    "mixed_10_20": (fill("rate", "m1"), unfilled("ambiguous", "m1")),
}


def source_rows():
    for name, digest in SOURCE_SHA256.items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"{name} не совпадает с ревизией {SOURCE_REVISION}")
    for split in SPLITS:
        for line in (SOURCE / "data" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            yield split, json.loads(line)


def convert(split, row):
    expected = row["expected"]
    if expected["document_type"] != "invoice" or row["currency"] != expected["currency"]:
        raise ValueError("Неожиданная разметка " + row["id"])
    rub = row["currency"] == "RUB"
    vat_mode, vat_rate = VAT[row["vat_mode"]]
    case, label = example(
        row["id"], SPLITS[split], "invoice_draft", [("user", INPUT_TEMPLATE.format(text=row["text"]))],
        recipient_name=fill(expected["buyer"]["name"], "m1"),
        recipient_inn=fill(expected["buyer"]["tax_id"], "m1"),
        # В счёте не в рублях цена и итог в другой валюте: рублёвых значений в документе нет.
        # Строка с отрицательной ценой — скидка: как в пайплайне, она не позиция и уже учтена в итоге.
        items=fill([item(i["description"], i["quantity"], i["unit"], i["unit_price"] if rub else None)
                    for i in expected["line_items"] if i["unit_price"] >= 0], "m1"),
        amount_rub=fill(expected["total_amount"], "m1") if rub else unfilled(),
        vat_mode=vat_mode, vat_rate_percent=vat_rate,
    )
    meta = {"case_id": row["id"], "source_split": split,
            **{key: row[key] for key in ("currency", "vat_mode", "template_family", "item_count")}}
    return case, label, meta


def build_contents():
    records = [convert(split, row) for split, row in source_rows()]
    for case, label, _ in records:
        validate_messages(InputCase.model_validate(case).messages)
        LABEL_ADAPTER.validate_python(label)
        for name, value in label["fields"].items():
            if value["status"] == "fillable" and invalid_format(name, value["value"]):
                raise ValueError(f"Некорректный формат эталона: {case['case_id']}.{name}")
    cases, labels, meta = zip(*records)
    encode = lambda rows: "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    inputs, targets = encode(cases), encode(labels)
    manifest = {
        "dataset_version": 1, "name": "ru_invoice", "synthetic": True,
        "authorship": "external benchmark, converted by scripts/build_ru_invoice_dataset.py",
        "source": {"name": "necrasov-ilya/ru-invoice-extraction-benchmark", "url": SOURCE_URL,
                   "revision": SOURCE_REVISION, "license": "Apache-2.0", "sha256": SOURCE_SHA256},
        "input_template": INPUT_TEMPLATE, "split_mapping": SPLITS,
        "external_fields": list(EXTERNAL_FIELDS), "placeholder_fields": list(PLACEHOLDER_FIELDS),
        "rules": {
            "recipient": "buyer of the source invoice",
            "non_rub": "amount_rub missing and unit_price_rub null; scored separately, not as accuracy",
            "discount_lines": "negative-price lines are discounts: not items, already included in total_amount",
            "vat": {key: [mode["value"], rate["status"], rate["value"]] for key, (mode, rate) in VAT.items()},
        },
        "case_count": len(cases),
        "input_sha256": hashlib.sha256(inputs.encode()).hexdigest(),
        "labels_sha256": hashlib.sha256(targets.encode()).hexdigest(),
        "cases": list(meta),
    }
    return {"cases_ru_invoice.jsonl": inputs, "labels_ru_invoice.jsonl": targets,
            "ru_invoice.manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, contents in build_contents().items():
        path = DATA / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != contents:
                raise SystemExit(f"{name} не соответствует scripts.build_ru_invoice_dataset")
        else:
            path.write_text(contents, encoding="utf-8")
    print("ru_invoice: 50 счетов внешнего бенчмарка (dev 40, test 10; 5 не в рублях). API не вызывались.")


if __name__ == "__main__":
    main()
