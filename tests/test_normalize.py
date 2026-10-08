"""Tests for normalize.py. Run: pytest"""

import pytest

from entity_resolution.normalize import (
    fold_to_ascii,
    manufacturer_in_name,
    model_tokens,
    normalize,
    numeric_tokens,
    parse_price,
    price_difference,
    tokenize,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Sony Turntable - PSLX350H", "sony turntable pslx350h"),
        ("  LG 2.0 cu.ft.  ", "lg 2.0 cu ft"),
        ("Café", "cafe"),
        ("BRAVIA® Sync", "bravia sync"),
        ("178° Wide Viewing Angle", "178 wide viewing angle"),
        ("", ""),
        (None, ""),
        (123, "123"),
        ("DVD±R/RW", "dvd r rw"),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_normalize_keeps_numbers():
    # Stripping digits would merge a 15 inch monitor with a 17 inch one.
    assert "15.4" in normalize("Notebook 15.4 inch")
    assert normalize("abc 123 def") != normalize("abc def")


def test_normalize_handles_soft_hyphen():
    # 0xAD in the real Abt.csv decodes to a soft hyphen under cp1252.
    assert normalize("Bluetooth\xad Ready") == "bluetooth ready"


def test_tokenize_keeps_model_numbers_whole():
    # Splitting pslx350h into pslx/350/h would discard the model number.
    assert tokenize("Sony PSLX350H 33-1/3") == ["sony", "pslx350h", "33", "1", "3"]


def test_numeric_tokens_keeps_decimals():
    assert numeric_tokens("LG 2.0 cu.ft. Microwave") == {"2.0"}


def test_model_tokens_mixes_letters_and_digits():
    assert "pslx350h" in model_tokens("Sony Turntable - PSLX350H")
    assert "ezxs88w" in model_tokens("Linksys EtherFast EZXS88W")


def test_model_tokens_includes_long_digit_runs():
    assert model_tokens("Cable 10ft Beige") == {"10ft"}
    assert "16000" in model_tokens("Contrast Ratio 10000:1 16000")


def test_model_tokens_excludes_pure_words_and_decimals():
    # "sony" is a brand, not a model; "2.0" is a capacity.
    assert "sony" not in model_tokens("Sony 2.0 cu.ft. Washer")
    assert "2.0" not in model_tokens("Sony 2.0 cu.ft. Washer")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("$399.00", 399.0),
        ("$1,000.00", 1000.0),
        ("$9.99", 9.99),
        ("", None),
        (None, None),
        ("not a price", None),
        ("-$5.00", None),
    ],
)
def test_parse_price(raw, expected):
    assert parse_price(raw) == expected


def test_price_difference_relative():
    assert price_difference(100.0, 100.0) == 0.0
    assert price_difference(100.0, 200.0) == 0.5
    assert price_difference(0.0, 0.0) == 0.0


def test_price_difference_missing_returns_zero():
    # Deliberate convention: paired with the price_known_both indicator so a
    # missing price never looks like an equal price.
    assert price_difference(None, 100.0) == 0.0
    assert price_difference(100.0, None) == 0.0
    assert price_difference(None, None) == 0.0


def test_manufacturer_in_name():
    assert manufacturer_in_name("Sony Turntable PSLX350H", "Sony") == 1.0
    assert manufacturer_in_name("Sony Turntable PSLX350H", "Bose") == 0.0
    assert manufacturer_in_name("", "Sony") == 0.0
    assert manufacturer_in_name("Sony Turntable", "") == 0.0


def test_manufacturer_in_name_is_case_insensitive():
    assert manufacturer_in_name("Bose Acoustimass 5", "BOSE") == 1.0


def test_fold_to_ascii_spaces_out_unmappable_symbols():
    # Deleting rather than spacing would fuse the neighbouring words.
    assert fold_to_ascii("®") == " "
    assert fold_to_ascii("DVD±R") == "DVD R"


def test_normalize_does_not_fuse_words_across_a_symbol():
    assert normalize("DVD±R/RW") == "dvd r rw"
    assert normalize("DVD±R/RW") != "dvdr rw"