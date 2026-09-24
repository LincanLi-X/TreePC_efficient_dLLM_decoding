from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


def extract_final_number(text: str) -> str | None:
    explicit = re.findall(r"Final answer:\s*([-+]?[$]?\d[\d,]*(?:\.\d+)?)", text, flags=re.I)
    candidates = explicit or re.findall(r"[-+]?[$]?\d[\d,]*(?:\.\d+)?", text)
    return candidates[-1].replace("$", "").replace(",", "") if candidates else None


def gsm8k_exact_match(prediction: str, reference: str) -> bool:
    value = extract_final_number(prediction)
    if value is None:
        return False
    try:
        return Decimal(value) == Decimal(str(reference).replace(",", ""))
    except InvalidOperation:
        return value == str(reference).strip()
