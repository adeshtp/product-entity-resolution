"""Text normalization for product names and descriptions.

Numbers are never stripped: model numbers ("PSLX350H"), capacities ("2.0 cu.ft.")
and sizes ("15.4") carry product identity, so removing digits would merge a 15 inch
monitor with a 17 inch one.

Unicode is folded to ASCII so "Café" and "Cafe" match. Symbols with no ASCII form
become a space rather than being deleted, because deleting them fuses the words on
either side ("DVD±R" would become "dvdr").
"""
import re
import unicodedata

_NUMBER = r"\d+(?:[.,]\d+)*"
# Numbers first, then any alphanumeric run kept whole: "pslx350h" must stay one
# token, because splitting it into pslx / 350 / h discards the model number.
# The lookahead stops the number branch claiming the digits of a mixed token, so
# "10ft" does not become "10" + "ft".
_TOKEN_RE = re.compile(rf"{_NUMBER}(?![a-z0-9])|[a-z0-9]+")
_MIXED_TOKEN_RE = re.compile(r"(?=.*[a-z])(?=.*\d)[a-z0-9]+")
_PURE_NUMBER_RE = re.compile(rf"^{_NUMBER}$")
_LONG_NUMBER_RE = re.compile(r"\d{3,}")

_WHITESPACE_RE = re.compile(r"\s+")


def fold_to_ascii(text: str) -> str:
    """Fold unicode to ASCII, e.g. 'Café' -> 'Cafe'.

    Characters with no ASCII form ('®', '±', '°') become a SPACE rather than
    being deleted. Deleting them would silently join the words on either side:
    'DVD±R/RW' would become 'dvdr' instead of 'dvd r'.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(ch if ord(ch) < 128 else " " for ch in decomposed)
    return ascii_only


def normalize(text: str | None) -> str:
    """Lowercase, fold to ASCII, keep numbers, collapse whitespace."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    folded = fold_to_ascii(text).lower()
    # Protect decimal points / commas that sit between digits.
    folded = re.sub(r"(?<=\d)[.,](?=\d)", "#", folded)
    cleaned = re.sub(r"[^a-z0-9#]+", " ", folded)
    cleaned = cleaned.replace("#", ".")
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def tokenize(text: str | None) -> list[str]:
    """Split normalized text into word and number tokens."""
    return _TOKEN_RE.findall(normalize(text))


def numeric_tokens(text: str | None) -> set[str]:
    """Pure numeric tokens, decimals kept as written.

    '2.0 cu.ft.' -> {'2.0'}. Capacity and size live in these tokens. Tokens like
    '10ft' are excluded here because they are model tokens, not measurements.
    """
    return {t for t in tokenize(text) if _PURE_NUMBER_RE.match(t)}


def model_tokens(text: str | None) -> set[str]:
    """Tokens that look like a specific model or part number.

    Two shapes qualify: an alphanumeric run containing both letters and digits
    (PSLX350H, EZXS88W, WRT160NL), and a run of three or more digits that is
    not part of a decimal (160, 16000). Pure alphabetic brand words are excluded
    because they are shared across many different products.
    """
    tokens = tokenize(text)
    found: set[str] = set()
    for token in tokens:
        if _PURE_NUMBER_RE.match(token):
            # A pure number: only long runs are part numbers, not "5" or "12".
            if _LONG_NUMBER_RE.fullmatch(token):
                found.add(token)
        elif _MIXED_TOKEN_RE.fullmatch(token):
            # Letters and digits together: PSLX350H, EZXS88W, 10ft.
            found.add(token)
    return found


def parse_price(value: str | None) -> float | None:
    """Parse '$1,000.00' into 1000.0. Returns None for empty or unparsable."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("$", "").replace(",", "").replace(" ", "")
    try:
        price = float(text)
    except ValueError:
        return None
    if price != price or price < 0:  # NaN or negative
        return None
    return price


def price_difference(price_a: float | None, price_b: float | None) -> float:
    """Relative price difference in [0, 1], or 0.0 when either price is missing.

    Returning 0.0 for missing is a deliberate convention, and it is always read
    together with the `price_known_both` indicator feature. A missing price must
    not look like "both products cost the same", and the indicator is what
    carries that distinction to the model.
    """
    if price_a is None or price_b is None:
        return 0.0
    higher = max(price_a, price_b)
    if higher <= 0:
        return 0.0
    return abs(price_a - price_b) / higher


def manufacturer_in_name(name_a: str | None, manufacturer_b: str | None) -> float:
    """1.0 when the Buy manufacturer string occurs in the normalized Abt name.

    Catalog A has no manufacturer column, so manufacturer agreement cannot be a
    symmetric comparison. This is the substitute: check the B-side manufacturer
    token sequence against the A-side product name.

    Limitations, stated plainly:
    - one-directional. It cannot detect the case where the Abt name omits the
      brand but the description carries it.
    - a short manufacturer string is easy to hit by accident ("GE", "SA"), so a
      match is weak evidence, not proof.
    - the 116 manufacturer values are not normalized against each other
      ("Sony" vs "SONY COMPUTER ENTERTAINME"), so this feature is often 0 for a
      correct pair. It measures evidence for a match, never evidence against.
    """
    brand = normalize(manufacturer_b)
    if not brand:
        return 0.0
    haystack = normalize(name_a)
    if not haystack:
        return 0.0
    return 1.0 if brand in haystack else 0.0