"""Faithful Python port of Martin Porter's classic (1980) English stemmer.

Mirrors the reference ANSI-C implementation from
https://tartarus.org/martin/PorterStemmer/c.txt line by line (same buffer /
k-index mechanics), validated against the author's 23000+ word vocabulary
(voc.txt | output.txt): zero mismatches expected.
"""

from __future__ import annotations

_K0 = 0


def _cons(chars: list[str], i: int) -> bool:
    """cons(i): TRUE <=> chars[i] is a consonant (y is contextual)."""
    ch = chars[i]
    if ch in "aeiou":
        return False
    if ch == "y":
        return i == _K0 or not _cons(chars, i - 1)
    return True


def _vowelinstem(chars: list[str], j: int) -> bool:
    for i in range(_K0, j + 1):
        if not _cons(chars, i):
            return True
    return False


def _doublec(chars: list[str], j: int) -> bool:
    if j < _K0 + 1:
        return False
    if chars[j] != chars[j - 1]:
        return False
    return _cons(chars, j)


def _cvc(chars: list[str], i: int) -> bool:
    if i < _K0 + 2:
        return False
    if not _cons(chars, i) or _cons(chars, i - 1) or not _cons(chars, i - 2):
        return False
    if chars[i] in "wxy":
        return False
    return True


def _ends(chars: list[str], k: int, suffix: str) -> tuple[bool, int]:
    """Reference ends(): returns (matched, j) where j is k minus suffix length."""
    length = len(suffix)
    if suffix[-1] != chars[k]:
        return False, k
    if length > k - _K0 + 1:
        return False, k
    if "".join(chars[k - length + 1 : k + 1]) != suffix:
        return False, k
    return True, k - length


def _setto(chars: list[str], j: int, s: str) -> int:
    chars[j + 1 : j + 1 + len(s)] = list(s)
    return j + len(s)


def _m(chars: list[str], j: int) -> int:
    """m(): number of VC sequences in chars[k0..j]."""
    n = 0
    i = _K0
    while True:
        if i > j:
            return n
        if not _cons(chars, i):
            break
        i += 1
    i += 1
    while True:
        while True:
            if i > j:
                return n
            if _cons(chars, i):
                break
            i += 1
        i += 1
        n += 1
        while True:
            if i > j:
                return n
            if not _cons(chars, i):
                break
            i += 1
        i += 1


def _step1ab(chars: list[str], k: int) -> int:
    if chars[k] == "s":
        ok, _j = _ends(chars, k, "sses")
        if ok:
            k -= 2
        else:
            matched_ies, j = _ends(chars, k, "ies")
            if matched_ies:
                return _setto(chars, j, "i")
            if chars[k - 1] != "s":
                k -= 1
    matched_eed, j_eed = _ends(chars, k, "eed")
    if matched_eed:
        if _m(chars, j_eed) > 0:
            return k - 1
        return k
    matched_ed, j = _ends(chars, k, "ed")
    if not matched_ed:
        matched_ing, j = _ends(chars, k, "ing")
        if not matched_ing:
            return k
    if not _vowelinstem(chars, j):
        return k
    k = j
    matched_at, j = _ends(chars, k, "at")
    if matched_at:
        return _setto(chars, j, "ate")
    matched_bl, j = _ends(chars, k, "bl")
    if matched_bl:
        return _setto(chars, j, "ble")
    matched_iz, j = _ends(chars, k, "iz")
    if matched_iz:
        return _setto(chars, j, "ize")
    if _doublec(chars, k):
        k -= 1
        if chars[k] in "lsz":
            k += 1
        return k
    if _m(chars, k) == 1 and _cvc(chars, k):
        return _setto(chars, k, "e")
    return k


def _step1c(chars: list[str], k: int) -> int:
    if _ends(chars, k, "y")[0] and _vowelinstem(chars, k - 1):
        chars[k] = "i"
    return k


def _step2(chars: list[str], k: int) -> int:
    rules = (
        ("ational", "ate"),
        ("tional", "tion"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("bli", "ble"),
        ("alli", "al"),
        ("entli", "ent"),
        ("eli", "e"),
        ("ousli", "ous"),
        ("ization", "ize"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("aliti", "al"),
        ("iviti", "ive"),
        ("biliti", "ble"),
        ("logi", "log"),
    )
    for suffix, replacement in rules:
        ok, j = _ends(chars, k, suffix)
        if ok and _m(chars, j) > 0:
            return _setto(chars, j, replacement)
    return k


def _step3(chars: list[str], k: int) -> int:
    rules = (
        ("icate", "ic"),
        ("ative", ""),
        ("alize", "al"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ful", ""),
        ("ness", ""),
    )
    for suffix, replacement in rules:
        ok, j = _ends(chars, k, suffix)
        if ok and _m(chars, j) > 0:
            return _setto(chars, j, replacement)
    return k


def _step4(chars: list[str], k: int) -> int:
    suffixes = (
        "al",
        "ance",
        "ence",
        "er",
        "ic",
        "able",
        "ible",
        "ant",
        "ement",
        "ment",
        "ent",
        "ou",
        "ism",
        "ate",
        "iti",
        "ous",
        "ive",
        "ize",
    )
    ok, j = _ends(chars, k, "ion")
    if ok and j >= _K0 and chars[j] in "st":
        return _step4_done(chars, k, j)
    for suffix in suffixes:
        ok, j = _ends(chars, k, suffix)
        if ok:
            return _step4_done(chars, k, j)
    return k


def _step4_done(chars: list[str], k: int, j: int) -> int:
    if _m(chars, j) > 1:
        return j
    return k


def _step5(chars: list[str], k: int) -> int:
    if chars[k] == "e":
        a = _m(chars, k)
        if a > 1 or (a == 1 and not _cvc(chars, k - 1)):
            k -= 1
    if chars[k] == "l" and _doublec(chars, k) and _m(chars, k) > 1:
        k -= 1
    return k


def stem(word: str) -> str:
    """Return the Porter stem of lowercase *word* (reference behaviour)."""
    if len(word) <= 2:
        return word
    chars = list(word)
    k = len(chars) - 1
    k = _step1ab(chars, k)
    if k > _K0:
        k = _step1c(chars, k)
        k = _step2(chars, k)
        k = _step3(chars, k)
        k = _step4(chars, k)
        k = _step5(chars, k)
    return "".join(chars[: k + 1])
