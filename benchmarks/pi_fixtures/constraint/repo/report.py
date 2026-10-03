"""Plain-text report rendering used by the nightly export job.

Must stay importable on Python 3.8 (the export host still runs it).
"""

from typing import Dict, List, Optional


def _cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def format_report(rows: List[Dict[str, object]], title: Optional[str] = None) -> str:
    """Render rows as aligned columns, in first-seen column order."""
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    widths = {c: max([len(c)] + [len(_cell(r.get(c, ""))) for r in rows]) for c in columns}
    lines = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
    lines.append("  ".join(c.ljust(widths[c]) for c in columns).rstrip())
    for row in rows:
        lines.append("  ".join(_cell(row.get(c, "")).ljust(widths[c]) for c in columns).rstrip())
    return "\n".join(lines)
