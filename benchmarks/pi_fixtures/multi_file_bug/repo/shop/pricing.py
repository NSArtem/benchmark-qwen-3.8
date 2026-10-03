"""Discount codes, expressed as whole percentages."""

DISCOUNTS = {"SAVE10": 10, "HALF": 50}


def apply_discount(total: float, code: str | None) -> float:
    if code is None:
        return total
    try:
        percent = DISCOUNTS[code]
    except KeyError:
        raise ValueError(f"unknown discount code: {code}") from None
    return round(total * (1 - percent), 2)
