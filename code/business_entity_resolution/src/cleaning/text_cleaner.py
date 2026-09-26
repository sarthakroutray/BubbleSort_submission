"""Linguistic normalization: anyascii → lower/fold → number norms → skeleton → core name.

Spec: FINAL_RESEARCH_AND_IMPLEMENTATION_PLAN.md §5.2. Country-agnostic: nothing
here branches on country and no country-specific vocabulary is hard-coded beyond
the generic legal-form / street-abbreviation lists that apply everywhere.

Two name views are emitted (raw + legal-suffix-stripped "core"); downstream
similarities are computed on both.
"""

import re
from functools import lru_cache

from anyascii import anyascii

_ASCII_ONLY = re.compile(r"\A[\x00-\x7f]*\Z")
_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_ZERO_BETWEEN_LETTERS = re.compile(r"(?<=[a-z])0(?=[a-z])")
_LETTER_DIGIT_BOUNDARY = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# --- documented folding substitutions (plan §5.2 step 2 / §3.4) ---
WORD_NUMERALS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
}

# Legal-form tokens stripped from the trailing end for the "core name" view.
LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "lc",
    "ltd", "limited", "lp", "llp", "lllp", "plc", "pllc", "pc", "pvt", "private",
    "pte", "gmbh", "mbh", "ag", "kg", "kgaa", "ohg", "eg", "ug",
    "sa", "sas", "sasu", "sarl", "eurl", "snc", "sci", "scp", "sca", "scs",
    "bv", "nv", "cv", "vof", "srl", "spa", "srls", "sro", "zrt", "kft",
    "oy", "oyj", "ab", "as", "asa", "aps", "a/s", "i/s", "kk", "gk", "yk",
    "pty", "sdn", "bhd", "ltda", "sl", "slu", "sac", "eirl", "sprl", "doo",
    "dd", "doo", "ooo", "oao", "zao", "pao", "jsc", "ojsc", "cjsc",
    "dba", "trading", "enterprises", "enterprise",
}

# Generic addressing-word canonicalization (fold, don't filter).
STREET_ABBR = {
    "street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "court": "ct", "place": "pl", "terrace": "ter",
    "highway": "hwy", "parkway": "pkwy", "circle": "cir", "square": "sq",
    "crescent": "cres", "gardens": "gdns", "grove": "grv",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "mount": "mt", "saint": "st", "fort": "ft",
    "apartment": "apt", "suite": "ste", "building": "bldg", "floor": "fl",
    "block": "blk", "sector": "sec", "phase": "ph", "district": "dist",
    "post": "po", "postal": "po", "number": "no", "num": "no",
    "rue": "r", "impasse": "imp", "chemin": "che", "route": "rte",
    "allee": "all", "quai": "q", "avenida": "ave", "calle": "cl",
    "straat": "str", "und": "and", "y": "and",
}


@lru_cache(maxsize=500_000)
def transliterate(text: str) -> str:
    """ASCII-fold with a pure-ASCII fast path; never returns None.

    anyascii maps Devanagari/Telugu/Cyrillic/accents to ASCII. Mandatory:
    6.2% of train / 7.4% of test S2/S3 names vanish under naive ASCII strip.
    """
    if not text:
        return ""
    if _ASCII_ONLY.match(text):
        return text
    out = anyascii(text)
    return out if out is not None else ""


@lru_cache(maxsize=500_000)
def normalize(text: str) -> str:
    """Lowercase, ordinal/zero/numeral fold, letter-digit split, zero-strip.

    Returns a space-separated token string of [a-z0-9] tokens. Country-agnostic.
    """
    if not text:
        return ""
    s = transliterate(text).lower()
    s = s.replace("&", " and ")
    s = _ORDINAL.sub(r"\1", s)
    s = _ZERO_BETWEEN_LETTERS.sub("o", s)
    s = _LETTER_DIGIT_BOUNDARY.sub(" ", s)
    out = []
    for tok in _NON_ALNUM.split(s):
        if not tok:
            continue
        mapped = WORD_NUMERALS.get(tok)
        if mapped is not None:
            out.append(mapped)
        elif tok.isdigit():
            out.append(tok.lstrip("0") or "0")
        else:
            out.append(tok)
    return " ".join(out)


def tokens(text: str) -> list:
    n = normalize(text)
    return n.split() if n else []


def canon_token(tok: str) -> str:
    return STREET_ABBR.get(tok, tok)


def name_views(name: str):
    """(raw_normalized, core_normalized) — core drops trailing legal forms."""
    raw = normalize(name)
    toks = raw.split()
    core = list(toks)
    while len(core) > 1 and core[-1] in LEGAL_SUFFIXES:
        core.pop()
    if not core and toks:
        core = list(toks)
    return raw, " ".join(core)


def address_words(address: str) -> list:
    """Canonicalized address tokens (street/road/direction folded)."""
    return [canon_token(t) for t in tokens(address)]


def numbers(text: str) -> list:
    """Digit tokens with leading zeros already stripped by normalize()."""
    return [t for t in tokens(text) if t.isdigit()]


def number_word_pairs(address: str) -> list:
    """House-number + following word, e.g. ``203 26th`` -> ``203_26th``."""
    toks = address_words(address)
    pairs = []
    for i, t in enumerate(toks[:-1]):
        if t.isdigit():
            pairs.append(f"{t}_{toks[i + 1]}")
    return pairs


@lru_cache(maxsize=500_000)
def skeleton(tok: str) -> str:
    """Phonetic consonant skeleton for one token (plan §5.2 step 4).

    Documented substitutions only: ph→f, bh/v/w→b, kh→k, th→t, gh→(silent),
    tion→sn, g/z→j, m→n, c→s before e/i/y else k, q→k, x→ks, drop vowels,
    collapse repeats. ``tirupti``/``tirupati`` -> ``trpt``; ``finance`` -> ``fnns``.
    """
    if not tok:
        return ""
    s = tok
    s = s.replace("tion", "sn").replace("sion", "sn")
    s = s.replace("ph", "f")
    s = s.replace("gh", "")
    s = s.replace("kh", "k")
    s = s.replace("bh", "b").replace("th", "t").replace("dh", "d")
    s = s.replace("sh", "s").replace("ch", "c")
    s = s.replace("ck", "k").replace("q", "k").replace("x", "ks")
    out = []
    for i, ch in enumerate(s):
        if ch in "aeiouh":
            continue
        if ch == "c":
            nxt = s[i + 1] if i + 1 < len(s) else ""
            out.append("s" if nxt in "eiy" else "k")
        elif ch in "gzj":
            out.append("j")
        elif ch in "vwb":
            out.append("b")
        elif ch == "m":
            out.append("n")
        elif ch == "y":
            out.append("y")
        elif ch.isalnum():
            out.append(ch)
    collapsed = []
    for ch in out:
        if not collapsed or collapsed[-1] != ch:
            collapsed.append(ch)
    return "".join(collapsed)


def skeleton_text(text: str) -> str:
    return " ".join(skeleton(t) for t in tokens(text))


if __name__ == "__main__":
    import sys
    for a in sys.argv[1:]:
        raw, core = name_views(a)
        print(f"{a!r}\n  raw  = {raw!r}\n  core = {core!r}\n  skel = {skeleton_text(a)!r}")
