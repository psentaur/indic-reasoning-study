"""Conservative numerical exact match and observable explanation diagnostics."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
import unicodedata

SCORER_VERSION = "1"


def normalize_number(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().replace("−", "-")
    text = "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)
    # Accept conventional western and Indian grouping, not arbitrary comma removal.
    if "," in text:
        if not re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})*,\d{3})(?:\.\d+)?", text):
            raise ValueError("Invalid digit grouping")
        text = text.replace(",", "")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", text):
        raise ValueError("Expected one finite decimal number without units")
    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise ValueError("Invalid number") from error
    if number == 0:
        return "0"
    rendered = format(number, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def extract_answer(response: str) -> tuple[str | None, str | None]:
    markers = re.findall(r"(?m)^\s*FINAL_ANSWER:\s*(.*?)\s*$", response)
    if not markers:
        return None, "missing_marker"
    if len(markers) != 1:
        return None, "multiple_markers"
    try:
        return normalize_number(markers[0]), None
    except ValueError:
        return None, "invalid_number"


def numbered_steps(response: str) -> list[str]:
    return re.findall(r"(?m)^\s*\d+[.)]\s+(.+)$", response)


def script_counts(text: str) -> dict[str, int]:
    counts = {"latin_letters": 0, "bengali_letters": 0, "telugu_letters": 0}
    for character in text:
        if not unicodedata.category(character).startswith("L"):
            continue
        point = ord(character)
        if 0x0980 <= point <= 0x09FF:
            counts["bengali_letters"] += 1
        elif 0x0C00 <= point <= 0x0C7F:
            counts["telugu_letters"] += 1
        elif "LATIN" in unicodedata.name(character, ""):
            counts["latin_letters"] += 1
    return counts


def score_response(row: dict, gold: str) -> dict:
    answer, parse_error = extract_answer(row["response"])
    steps = numbered_steps(row["response"])
    explanation = "\n".join(steps)
    scripts = script_counts(explanation)
    target = {"en": "latin_letters", "bn": "bengali_letters", "te": "telugu_letters"}[row["reasoning_language"]]
    letters = sum(scripts.values())
    return {
        **row, "scorer_version": SCORER_VERSION, "answer": answer,
        "correct": answer is not None and answer == normalize_number(gold),
        "parse_error": parse_error, "actual_steps": len(steps),
        "step_compliant": len(steps) == row["budget"],
        "repeated_step": len(set(s.strip().casefold() for s in steps)) < len(steps),
        **scripts,
        "target_script_fraction": scripts[target] / letters if letters else None,
        "script_check_note": "Observable script counts only; not a language or semantic correctness judgment"
    }
