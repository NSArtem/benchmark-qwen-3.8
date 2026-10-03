"""Exhaustive conversion check with a verbose log (prints one line per number)."""

from roman import from_roman, to_roman

REFERENCE = {4: "IV", 9: "IX", 14: "XIV", 40: "XL", 90: "XC", 400: "CD", 900: "CM",
             1994: "MCMXCIV", 2024: "MMXXIV", 3999: "MMMCMXCIX", 944: "CMXLIV"}

failures = []
for number in list(range(1, 200)) + [900, 944, 999, 1994, 2024, 3999]:
    text = to_roman(number)
    back = from_roman(text)
    expected = REFERENCE.get(number)
    ok = back == number and (expected is None or expected == text)
    print(f"{number:5d} -> {text:<16} -> {back:5d}  {'ok' if ok else 'MISMATCH'}")
    if not ok:
        failures.append((number, text, back, expected))

print()
if failures:
    print(f"{len(failures)} failures:")
    for number, text, back, expected in failures:
        print(f"  {number}: got {text!r} (round trip {back}), expected {expected!r}")
    raise SystemExit(1)
print("all conversions ok")
