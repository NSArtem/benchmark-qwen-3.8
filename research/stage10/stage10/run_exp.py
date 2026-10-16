"""Run one Phase B experiment end to end (TASK.md section 4 procedure)."""

from __future__ import annotations

import argparse
import datetime
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .build import CANDIDATES, EXPERIMENTS
from .candidate import PRODUCTION
from .common import host_check, wait_for_preflight
from .reference import RESULTS

STAGE10 = Path(__file__).resolve().parents[1]
LOG = STAGE10 / "LOG.md"

# exp_id -> (recipe, loader, model path, rotate dtype, has build)
SPECS = {
    "E01": ("prod-g128", "mlx_lm", PRODUCTION, None, False),
    "E05": ("had3-rotbf16", "candidate", CANDIDATES / "E03", "bfloat16", False),
}
for _exp, _recipe in EXPERIMENTS.items():
    SPECS[_exp] = (_recipe, "candidate", CANDIDATES / _exp, None, True)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=STAGE10, capture_output=True, text=True, check=True
    ).stdout.strip()


def log_entry(title: str, cmd: list[str], result: str, run: dict | None, facts: str, peak=None) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    wall = run.get("wall_s") if run else None
    wall_text = f"{wall / 60:.1f}m" if wall is not None else "n/a"
    peak_text = f"{peak:.2f} GiB" if isinstance(peak, (int, float)) else "n/a"
    swap = (
        f"{run['swap_mib_before'] / 1024:.2f}/{run['swap_mib_after'] / 1024:.2f} GiB"
        if run and "swap_mib_before" in run
        else "n/a"
    )
    entry = (
        f"\n## {now} {title}\n"
        f"cmd: `{shlex.join(cmd)}`\n"
        f"commit: {git_commit()}\n"
        f"result: {result}\n"
        f"wall: {wall_text}, peak_mlx: {peak_text}, swap before/after: {swap}\n"
        f"facts: {facts}\n"
    )
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(entry)


def _run_step(module: str, args: list[str]) -> tuple[int, float]:
    cmd = [sys.executable, "-m", module, *args]
    started = time.monotonic()
    code = subprocess.run(cmd, cwd=STAGE10).returncode
    return code, time.monotonic() - started


def _read(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def step_result(exit_code: int, run: dict | None) -> str:
    if run and run.get("safety_stop"):
        return "FAILED_MEMORY (watchdog SAFETY STOP)"
    return "OK" if exit_code == 0 else f"FAILED_ERROR (exit {exit_code})"


def _tail_error(log_path: str | None) -> str:
    if not log_path or not Path(log_path).is_file():
        return ""
    lines = [
        line
        for line in Path(log_path).read_text(errors="replace").splitlines()
        if line.strip() and " free=" not in line and not line.startswith("elapsed_s")
    ]
    return lines[-1][:300] if lines else ""


def run_build(exp: str) -> bool:
    args = ["--exp", exp]
    code, _ = _run_step("stage10.build", args)
    run = _read(RESULTS / exp / "build_run.json")
    build = _read(RESULTS / exp / "build.json") or {}
    ck = _read(RESULTS / exp / "checkpoint.json") or {}
    facts = (
        f"checkpoint {ck.get('path')}, bytes {ck.get('bytes')}, gib {ck.get('gib')}, bpw {ck.get('bpw')}; "
        f"build wall {build.get('wall_s')} s; min free {run and run.get('free_pct_min')}%"
        if code == 0
        else f"error: {_tail_error(run and run.get('log_path'))}"
    )
    log_entry(f"{exp} build", ["uv", "run", "python", "-m", "stage10.build", *args], step_result(code, run), run, facts, build.get("peak_mlx_gib"))
    return code == 0


def run_eval(exp: str, model_args: list[str]) -> bool:
    code, _ = _run_step("stage10.evaluate", model_args)
    run = _read(RESULTS / exp / "eval_run.json")
    ev = _read(RESULTS / exp / "eval.json") or {}
    if code == 0:
        facts = "; ".join(
            f"{part}: nll {ev[part]['nll_mean']:.5f}, delta {ev[part]['delta_nll']:.5f}, "
            f"top1 {ev[part]['top1_agree']:.4f}, kl_top32 {ev[part]['kl_top32']:.5f}"
            for part in ("all", "en", "code", "ru")
        )
    else:
        facts = f"error: {_tail_error(run and run.get('log_path'))}"
    log_entry(f"{exp} eval", ["uv", "run", "python", "-m", "stage10.evaluate", *model_args], step_result(code, run), run, facts, (ev.get("_run") or {}).get("peak_mlx_gib"))
    return code == 0


def run_gen(exp: str, model_args: list[str]) -> bool:
    code, _ = _run_step("stage10.gen_check", model_args)
    run = _read(RESULTS / exp / "gen_run.json")
    gen = _read(RESULTS / exp / "gen_check.json") or []
    if code == 0:
        facts = "; ".join(
            f"{g['id']}: n {g['n_tokens']}, eos {g['hit_eos']}, distinct4 {g['distinct4']:.3f}, max4 {g['max4']}"
            + (f", has_187 {g['p2_has_187']}" if "p2_has_187" in g else "")
            + (f", json_valid {g['p4_json_valid']}" if "p4_json_valid" in g else "")
            for g in gen
        )
    else:
        facts = f"error: {_tail_error(run and run.get('log_path'))}"
    log_entry(f"{exp} gen-check", ["uv", "run", "python", "-m", "stage10.gen_check", *model_args], step_result(code, run), run, facts)
    return code == 0


def run_ladder(exp: str, model_args: list[str], weights_gib: float) -> bool:
    args = [*model_args, "--weights-gib", f"{weights_gib:.4f}"]
    code, wall = _run_step("stage10.ladder", args)
    rows = _read(RESULTS / exp / "ladder.json") or []
    facts = "; ".join(
        f"ctx {r['ctx']}: decode {r.get('decode_tok_s')}, prefill {r.get('prefill_tok_s')} tok/s, "
        f"ttft {r.get('ttft_s')} s, peak {r.get('peak_mlx_gib')} GiB, min free {r.get('free_pct_min')}%, "
        f"safety_stop {r['safety_stop']}, exit {r['exit_code']}"
        for r in rows
    )
    stops = [r for r in rows if r["safety_stop"]]
    result = "OK" if code == 0 and not stops and all(r["exit_code"] == 0 for r in rows) else (
        "FAILED_MEMORY (watchdog SAFETY STOP)" if stops else f"FAILED_ERROR (exit {code})"
    )
    run = {"wall_s": wall}
    if rows:
        run.update(swap_mib_before=rows[0]["swap_mib_before"], swap_mib_after=rows[-1]["swap_mib_after"])
    peaks = [r.get("peak_mlx_gib") for r in rows if r.get("peak_mlx_gib") is not None]
    log_entry(f"{exp} ladder", ["uv", "run", "python", "-m", "stage10.ladder", *args], result, run, facts, max(peaks) if peaks else None)
    return result == "OK"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, choices=sorted(SPECS))
    parser.add_argument("--steps", default="build,eval,gen,ladder")
    args = parser.parse_args()
    recipe, loader, path, rotate_dtype, has_build = SPECS[args.exp]
    steps = args.steps.split(",")
    wait_for_preflight()
    model_args = ["--exp", args.exp, "--model", str(path), "--loader", loader]
    if rotate_dtype:
        model_args += ["--rotate-dtype", rotate_dtype]
    status = {}
    if has_build and "build" in steps:
        status["build"] = run_build(args.exp)
        if not status["build"]:
            print(json.dumps(status))
            sys.exit(1)
    weights_gib = None
    ck = _read(RESULTS / args.exp / "checkpoint.json")
    if args.exp == "E05":
        ck = _read(RESULTS / "E03" / "checkpoint.json")
    if args.exp == "E01":
        from .candidate import checkpoint_size

        ck = checkpoint_size(PRODUCTION)
        (RESULTS / args.exp).mkdir(parents=True, exist_ok=True)
        (RESULTS / args.exp / "checkpoint.json").write_text(json.dumps(ck, indent=2) + "\n")
    if ck:
        weights_gib = ck["gib"]
        if args.exp == "E05":
            (RESULTS / args.exp).mkdir(parents=True, exist_ok=True)
            (RESULTS / args.exp / "checkpoint.json").write_text(json.dumps(ck, indent=2) + "\n")
    if "eval" in steps:
        status["eval"] = run_eval(args.exp, model_args)
    if "gen" in steps:
        status["gen"] = run_gen(args.exp, model_args)
    if "ladder" in steps:
        if weights_gib is None:
            raise SystemExit("no checkpoint size recorded; cannot choose ladder lengths")
        status["ladder"] = run_ladder(args.exp, model_args, weights_gib)
    after = host_check()
    print(json.dumps({"exp": args.exp, "recipe": recipe, "steps": status, "free_pct": after.free_pct}))


if __name__ == "__main__":
    main()
