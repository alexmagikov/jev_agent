"""Сложная выборка challenge: согласованность генератора и эталонов; без API."""

import json

import pytest

import scripts.build_challenge_dataset as builder


@pytest.fixture(scope="module")
def dataset():
    contents = builder.build_contents()
    cases = {row["case_id"]: row for row in map(json.loads, contents["cases_challenge.jsonl"].splitlines())}
    labels = {row["case_id"]: row for row in map(json.loads, contents["labels_challenge.jsonl"].splitlines())}
    return cases, labels, json.loads(contents["challenge.manifest.json"])


def test_data_files_match_the_generator():
    for name, contents in builder.build_contents().items():
        assert (builder.DATA / name).read_text(encoding="utf-8") == contents, name


def test_dev_and_test_cases_await_human_review(dataset):
    cases, _, manifest = dataset
    assert list(cases) == [f"H{i:03}" for i in range(1, 105)]
    # Длинные диалоги добавлены после H090, чтобы прежние ID не сдвинулись.
    assert [case["split"] for case in cases.values()] == ["dev"] * 30 + ["test"] * 60 + ["dev"] * 4 + ["test"] * 10
    assert manifest["human_reviewed"] is True and manifest["splits"] == {"dev": 34, "test": 70}


def test_long_dialogues_spread_data_and_keep_assistant_out_of_evidence(dataset):
    cases, labels, manifest = dataset
    long_ids = [meta["case_id"] for meta in manifest["cases"] if "long_dialogue" in meta["tags"]]
    assert long_ids == [f"H{i:03}" for i in range(91, 105)]
    assert all(len(cases[case_id]["messages"]) >= 9 for case_id in long_ids)
    for case_id in long_ids:
        roles = {message["id"]: message["role"] for message in cases[case_id]["messages"]}
        assert all(roles[mid] == "user" for field in labels[case_id]["fields"].values() for mid in field["evidence"])
    summary = next(case_id for case_id in long_ids if "assistant_wrong_summary" in
                   next(meta["tags"] for meta in manifest["cases"] if meta["case_id"] == case_id))
    wrong = cases[summary]["messages"][1]["text"]
    gold = labels[summary]["fields"]["items"]["value"]
    # Пересказ ассистента называет неверную цену второй позиции; эталон берёт цену пользователя.
    assert str(int(gold[1]["unit_price_rub"])) not in wrong.replace(" ", "")


def test_no_two_cases_share_an_opening_message(dataset):
    openings = [case["messages"][0]["text"] for case in dataset[0].values()]
    assert len(set(openings)) == len(openings)


def test_generated_parameters_are_varied(dataset):
    # Раньше выбор по модулю давал 5 покупателей из 35 и один банк на семейство.
    _, labels, _ = dataset
    names = {label["fields"]["recipient_name"]["value"] for label in labels.values()
             if label["fields"].get("recipient_name", {}).get("status") == "fillable"}
    banks = {label["fields"]["destination_bank"]["value"] for label in labels.values()
             if label["fields"].get("destination_bank", {}).get("status") == "fillable"}
    assert len(names) >= 25 and len(banks) >= 6


def test_every_line_total_matches_the_gold_amount_except_partial_discounts(dataset):
    _, labels, manifest = dataset
    tags = {meta["case_id"]: meta["tags"] for meta in manifest["cases"]}
    checked = 0
    for case_id, label in labels.items():
        fields = label["fields"]
        lines = fields.get("items", {}).get("value") or []
        if {"partial_discount", "extra_charge"} & set(tags[case_id]) or not lines or any(
                line["quantity"] is None or line["unit_price_rub"] is None for line in lines):
            continue
        assert sum(line["quantity"] * line["unit_price_rub"] for line in lines) == fields["amount_rub"]["value"], case_id
        checked += 1
    assert checked >= 40


def test_generated_requisites_have_valid_check_digits(dataset):
    for label in dataset[1].values():
        fields = label["fields"]
        if fields.get("recipient_inn", {}).get("value"):
            assert builder.valid_inn(fields["recipient_inn"]["value"])
            assert builder.valid_account(fields["recipient_account"]["value"], fields["recipient_bik"]["value"])


@pytest.mark.parametrize("inn,valid", [("7810543216", True), ("7705123452", True), ("7810543217", False),
                                       ("771234567890", False)])
def test_inn_check_digits(inn, valid):
    assert builder.valid_inn(inn) is valid


def test_account_key_depends_on_bik():
    assert builder.valid_account("40702810200000067890", "044030653")
    assert not builder.valid_account("40702810200000067890", "044525225")


def test_seller_requisites_are_valid_decoys_but_not_the_buyer(dataset):
    cases, labels, _ = dataset
    fields = labels["H003"]["fields"]
    assert builder.valid_inn("7705123452") and builder.valid_account("40702810200000012345", "044525225")
    assert "7705123452" in cases["H003"]["messages"][0]["text"]
    assert (fields["recipient_inn"]["value"], fields["recipient_inn"]["evidence"]) == ("7810543216", ["m3"])


@pytest.mark.parametrize("case_id,amount", [("H001", 5035), ("H002", 37200), ("H006", 7400)])
def test_totals_equal_the_sum_of_current_lines(dataset, case_id, amount):
    fields = dataset[1][case_id]["fields"]
    lines = fields["items"]["value"]
    assert sum(line["quantity"] * line["unit_price_rub"] for line in lines) == fields["amount_rub"]["value"] == amount


def test_partial_discount_changes_only_the_total(dataset):
    fields = dataset[1]["H004"]["fields"]
    assert [line["unit_price_rub"] for line in fields["items"]["value"]] == [12400, 3100]
    assert fields["amount_rub"]["value"] == 3 * 12400 * 90 // 100 + 5 * 3100 == 48980


def test_outdated_and_foreign_values_are_not_gold(dataset):
    _, labels, _ = dataset
    assert labels["H005"]["fields"]["amount_rub"] | {"evidence": ["m3"]} == labels["H005"]["fields"]["amount_rub"]
    assert labels["H005"]["fields"]["amount_rub"]["value"] == 22500
    assert labels["H007"]["fields"]["destination_card_masked"]["value"] == "****7710"
    assert labels["H008"]["fields"]["amount_rub"]["value"] == 6400
    assert labels["H009"]["fields"]["amount_rub"]["status"] == "missing"
    assert labels["H010"]["document_type"] == "none"
