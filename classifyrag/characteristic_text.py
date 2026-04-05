"""
Reduce raw page/OCR text to document-type cues: titles and field labels, not filled values.

Typical forms use lines like "2. Tên khách hàng: Nguyễn ..." — we keep "Tên khách hàng"
and drop variable amounts, names, and dates so the text branch matches on layout/type.
"""

from __future__ import annotations

import re
from typing import Iterable

# Leading "1. " / "12. " on a line
_LINE_NUMBER_PREFIX = re.compile(r"^\s*\d+\.\s*")


def _norm_key(s: str) -> str:
    return " ".join(s.split()).casefold()


def _is_junk_label(s: str) -> bool:
    if len(s) < 2:
        return True
    if s.isdigit():
        return True
    # pure amount-like token
    if re.fullmatch(r"[\d.,\sVND]+", s, re.I):
        return True
    return False


def extract_characteristic_labels(text: str, *, max_phrases: int = 64) -> str:
    """
    From Vietnamese form-like text, collect title lines and text before ':' (field labels).

    - Numbered lines: strip ``N. `` then take the part before the first ``:`` if present,
      else the whole line (e.g. document title).
    - Deduplicate while preserving order (case-insensitive).
    - Result is comma-separated for ColSmol query/prototype text.
    """
    if not (text and text.strip()):
        return ""

    seen: set[str] = set()
    out: list[str] = []

    def push(phrase: str) -> None:
        p = " ".join(phrase.split())
        if _is_junk_label(p):
            return
        k = _norm_key(p)
        if k in seen:
            return
        seen.add(k)
        out.append(p)
        if len(out) >= max_phrases:
            return

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = _LINE_NUMBER_PREFIX.sub("", line, count=1).strip()
        if not line:
            continue
        if ":" in line:
            label = line.split(":", 1)[0].strip()
            push(label)
        else:
            push(line)

    if not out:
        return ""

    # If OCR collapsed everything into one line, try comma-split as fallback
    if len(out) == 1 and "," in out[0] and "\n" not in text.strip():
        single = out[0]
        out.clear()
        seen.clear()
        for part in _split_comma_phrases(single):
            push(part)

    return ", ".join(out)


def _split_comma_phrases(s: str) -> Iterable[str]:
    # Light split; avoid breaking "1,000,000"
    for part in s.split(","):
        t = part.strip()
        if t:
            yield t


def apply_characteristic_text(raw: str, enabled: bool) -> str:
    if not enabled or not (raw and raw.strip()):
        return raw
    reduced = extract_characteristic_labels(raw)
    return reduced if reduced else raw
