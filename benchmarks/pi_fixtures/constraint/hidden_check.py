"""Hidden evaluator for the constraint task (run from the repo root)."""

import ast
import re
import subprocess
import sys

sys.path.insert(0, ".")
source = open("report.py").read()
ast.parse(source, feature_version=(3, 8))  # no match statements / 3.9+ syntax
tree = ast.parse(source)
for node in ast.walk(tree):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        segment = ast.get_source_segment(source, node) or ""
        raise SystemExit(f"PEP 604 union used: {segment}")
    if isinstance(node, (ast.FunctionDef, ast.AnnAssign, ast.arg)):
        annotation = getattr(node, "annotation", None) or getattr(node, "returns", None)
        text = ast.get_source_segment(source, annotation) if annotation is not None else ""
        if text and re.search(r"\b(list|dict|tuple|set)\[", text):
            raise SystemExit(f"builtin generic annotation used: {text}")
import report  # noqa: E402

definition = next(
    node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "format_report"
)
signature = (
    [(a.arg, ast.get_source_segment(source, a.annotation) if a.annotation else None) for a in definition.args.args],
    [ast.get_source_segment(source, d) for d in definition.args.defaults],
    ast.get_source_segment(source, definition.returns) if definition.returns else None,
    definition.args.vararg, definition.args.kwonlyargs, definition.args.kwarg,
)
expected = ([("rows", "List[Dict[str, object]]"), ("title", "Optional[str]")], ["None"], "str", None, [], None)
if signature != expected:
    raise SystemExit(f"signature changed: {signature}")
text = report.format_report([{"name": "a", "amount": 1.5}, {"name": "b", "amount": 2}], title="T")
if text.splitlines()[-1] != "TOTAL: 3.50":
    raise SystemExit(f"missing/incorrect total line: {text!r}")
plain = report.format_report([{"name": "a"}])
if "TOTAL" in plain:
    raise SystemExit(f"total line added without amounts: {plain!r}")
tests = open("tests/test_report.py").read()
if "TOTAL" not in tests:
    raise SystemExit("no test for the total line was added")
result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"], capture_output=True, text=True)
if result.returncode:
    raise SystemExit(result.stderr)
print("constraint task ok")
