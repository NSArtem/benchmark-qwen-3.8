"""Autonomous, resumable stage runner for the Qwen3.8 overnight exploration."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "qwen38-overnight"
HARNESS = OUT / "harness"
REGISTRY = OUT / "results.json"
ENVIRONMENT = OUT / "environment.json"
METADATA = OUT / "metadata"
PROMPT = OUT / "coding-prompt.txt"

MODELS = {
    "q38-3bit": {
        "target": "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly",
        "mtp": "lukaskremla/Qwen3.8-27B-MTP-3bit-MLX",
        "metadata": METADATA / "lukaskremla__Qwen3.8-27B-3bit-MLX-TextOnly.json",
    },
    "q38-mixed-3.5": {
        "target": "rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX",
        "mtp": "lukaskremla/Qwen3.8-27B-MTP-3bit-MLX",
        "metadata": METADATA / "rapid-mlx__Qwen3.8-27B-mixed-3.5bpw-MLX.json",
    },
    "q38-2bit": {
        "target": "lukaskremla/Qwen3.8-27B-2bit-MLX-TextOnly",
        "mtp": "lukaskremla/Qwen3.8-27B-MTP-2bit-MLX",
        "metadata": METADATA / "lukaskremla__Qwen3.8-27B-2bit-MLX-TextOnly.json",
    },
}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def registry() -> dict[str, Any]:
    value = read_json(REGISTRY)
    return value if isinstance(value, dict) else {"schema_version": 1, "runs": [], "quality": []}


def append_command(command: list[str], note: str | None = None) -> None:
    with (OUT / "commands.txt").open("a", encoding="utf-8") as handle:
        if note:
            handle.write(f"# {note}\n")
        handle.write(shlex.join(command) + "\n")


def run_command(command: list[str], timeout: float, label: str, record: bool = True) -> int:
    if record:
        append_command(command, label)
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = ".uv-cache"
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        return 124


def child_group_from_watchdog_log(path: Path) -> int | None:
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get("event") == "watchdog_start":
                return int(value.get("child_pid"))
    except OSError:
        pass
    return None


def existing_run(candidate: str, stage: str) -> dict[str, Any] | None:
    for run in registry().get("runs", []):
        if run.get("candidate") == candidate and run.get("stage") == stage:
            return run
    return None


def healthy(run: dict[str, Any] | None, conservative: bool = False) -> bool:
    if not run:
        return False
    result = run.get("result", {})
    supervisor = run.get("supervisor", {})
    metrics = run.get("metrics", {})
    if result.get("status") != "complete" or supervisor.get("watchdog_abort"):
        return False
    if metrics.get("paging_assessment") in {"meaningful_sustained_paging", "severe_sustained_paging"}:
        return False
    if conservative and (metrics.get("minimum_free_memory_percent") or 0) < 20:
        return False
    return True


def commit(message: str) -> None:
    # Stage only files owned by this overnight task; unrelated user files and
    # archives at repository root are deliberately excluded.
    subprocess.run(["git", "add", "--", ".gitignore", "qwen38-overnight"], cwd=ROOT, check=False)
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=False)


def refresh_report() -> None:
    command = [
        "uv",
        "run",
        "python",
        "qwen38-overnight/harness/render_report.py",
        "--registry",
        "qwen38-overnight/results.json",
        "--environment",
        "qwen38-overnight/environment.json",
        "--output",
        "qwen38-overnight/results.md",
    ]
    run_command(command, 120, "regenerate results.md", record=False)


def record_stage(candidate: str, stage: str, result_path: Path, supervisor_path: Path, metadata_path: Path) -> None:
    command = [
        "uv",
        "run",
        "python",
        "qwen38-overnight/harness/record_stage.py",
        "--registry",
        "qwen38-overnight/results.json",
        "--candidate",
        candidate,
        "--stage",
        stage,
        "--result",
        str(result_path.relative_to(ROOT)),
        "--supervisor",
        str(supervisor_path.relative_to(ROOT)),
        "--metadata",
        str(metadata_path.relative_to(ROOT)),
    ]
    run_command(command, 120, f"record {candidate}/{stage}", record=False)
    refresh_report()


def recover(stage_dir: Path, stem: str) -> None:
    command = [
        "uv",
        "run",
        "python",
        "qwen38-overnight/harness/recover.py",
        "--log",
        str((stage_dir / f"recovery-{stem}.log").relative_to(ROOT)),
        "--max-wait",
        "120",
        "--settle-seconds",
        "8",
    ]
    run_command(command, 150, f"recover memory after {stage_dir.name}/{stem}", record=False)


def model_stage(
    candidate: str,
    stage: str,
    model: str,
    mode: str,
    phase: str,
    timeout: float,
    max_tokens: int = 128,
) -> dict[str, Any] | None:
    old = existing_run(candidate, stage)
    if old:
        return old
    stage_dir = OUT / candidate
    stage_dir.mkdir(parents=True, exist_ok=True)
    stem = {
        "target-tiny": "target-tiny",
        "target-practical": "target",
        "mtp-tiny": "mtp-tiny",
        "mtp-practical": "mtp",
    }.get(stage, stage)
    child_log = stage_dir / f"{stem}.log"
    memory_log = stage_dir / f"memory-{stem}.log"
    supervisor_json = stage_dir / f"{stem}.supervisor.json"
    child_result = stage_dir / f"{stem}.json"
    progress_file = stage_dir / f"{stem}.progress"
    text_output = stage_dir / f"{stem}-output.txt"
    tokens_output = stage_dir / f"{stem}-tokens.json"
    formatted_output = stage_dir / f"{stem}-formatted-prompt.txt"
    watchdog_command = [
        "uv",
        "run",
        "python",
        "qwen38-overnight/harness/watchdog.py",
        "--label",
        f"{candidate}:{stage}",
        "--memory-log",
        str(memory_log.relative_to(ROOT)),
        "--child-log",
        str(child_log.relative_to(ROOT)),
        "--result-json",
        str(supervisor_json.relative_to(ROOT)),
        "--child-result",
        str(child_result.relative_to(ROOT)),
        "--progress-file",
        str(progress_file.relative_to(ROOT)),
        "--timeout",
        str(timeout),
        "--",
        "uv",
        "run",
        "python",
        "qwen38-overnight/harness/run_model.py",
        "--model",
        model,
        "--mode",
        mode,
        "--phase",
        phase,
        "--prompt",
        str(PROMPT.relative_to(ROOT)),
        "--result-json",
        str(child_result.relative_to(ROOT)),
        "--text-output",
        str(text_output.relative_to(ROOT)),
        "--tokens-output",
        str(tokens_output.relative_to(ROOT)),
        "--formatted-output",
        str(formatted_output.relative_to(ROOT)),
        "--progress-file",
        str(progress_file.relative_to(ROOT)),
        "--max-tokens",
        str(max_tokens),
    ]
    draft = MODELS[candidate].get("mtp") if mode == "mtp" else None
    if draft:
        separator = watchdog_command.index("--")
        insert_at = watchdog_command.index("--model", separator + 1)
        watchdog_command[insert_at:insert_at] = ["--draft-model", draft]
    timeout_result = run_command(watchdog_command, timeout + 90, f"run {candidate}/{stage}")
    # If the outer runner timed out, clean up the exact child PID recorded by
    # the watchdog. This remains scoped to a process group we started.
    if timeout_result == 124:
        child_pid = child_group_from_watchdog_log(memory_log)
        if child_pid:
            try:
                os.killpg(child_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    recover(stage_dir, stem)
    record_stage(candidate, stage, child_result, supervisor_json, MODELS[candidate]["metadata"])
    commit(f"benchmark: record {candidate} {stage} results")
    return existing_run(candidate, stage)


def apc_stage(candidate: str, model: str, mode: str, port: int) -> dict[str, Any] | None:
    old = existing_run(candidate, "apc")
    if old:
        return old
    stage_dir = OUT / candidate
    stage_dir.mkdir(parents=True, exist_ok=True)
    result_path = stage_dir / "apc.json"
    supervisor_path = stage_dir / "apc.supervisor.json"
    command = [
        "uv",
        "run",
        "python",
        "qwen38-overnight/harness/apc_test.py",
        "--model",
        model,
        "--mode",
        mode,
        "--prompt",
        str(PROMPT.relative_to(ROOT)),
        "--output-dir",
        str(stage_dir.relative_to(ROOT)),
        "--result-json",
        str(result_path.relative_to(ROOT)),
        "--supervisor-json",
        str(supervisor_path.relative_to(ROOT)),
        "--port",
        str(port),
        "--timeout",
        "1500",
    ]
    if mode == "mtp":
        command.extend(["--draft-model", MODELS[candidate]["mtp"]])
    run_command(command, 1650, f"run {candidate}/apc")
    recover(stage_dir, "apc")
    record_stage(candidate, "apc", result_path, supervisor_path, MODELS[candidate]["metadata"])
    commit(f"benchmark: record {candidate} APC results")
    return existing_run(candidate, "apc")


def quality_stage(candidate: str, model: str) -> dict[str, Any] | None:
    old = next((item for item in registry().get("quality", []) if item.get("candidate") == candidate), None)
    if old:
        return old
    stage_dir = OUT / "quality"
    stage_dir.mkdir(parents=True, exist_ok=True)
    stem = candidate.replace("/", "-")
    child_log = stage_dir / f"{stem}.log"
    memory_log = stage_dir / f"memory-{stem}.log"
    supervisor_path = stage_dir / f"{stem}.supervisor.json"
    child_result = stage_dir / f"{stem}.json"
    progress_file = stage_dir / f"{stem}.progress"
    raw_output = stage_dir / f"{stem}-outputs.json"
    command = [
        "uv",
        "run",
        "python",
        "qwen38-overnight/harness/watchdog.py",
        "--label",
        f"quality:{candidate}",
        "--memory-log",
        str(memory_log.relative_to(ROOT)),
        "--child-log",
        str(child_log.relative_to(ROOT)),
        "--result-json",
        str(supervisor_path.relative_to(ROOT)),
        "--child-result",
        str(child_result.relative_to(ROOT)),
        "--progress-file",
        str(progress_file.relative_to(ROOT)),
        "--timeout",
        "1200",
        "--",
        "uv",
        "run",
        "python",
        "qwen38-overnight/harness/quality_runner.py",
        "--model",
        model,
        "--candidate",
        candidate,
        "--result-json",
        str(child_result.relative_to(ROOT)),
        "--raw-output",
        str(raw_output.relative_to(ROOT)),
        "--progress-file",
        str(progress_file.relative_to(ROOT)),
    ]
    run_command(command, 1290, f"quality smoke {candidate}")
    recover(stage_dir, f"{stem}-quality")
    record_command = [
        "uv",
        "run",
        "python",
        "qwen38-overnight/harness/record_quality.py",
        "--registry",
        str(REGISTRY.relative_to(ROOT)),
        "--candidate",
        candidate,
        "--result",
        str(child_result.relative_to(ROOT)),
        "--supervisor",
        str(supervisor_path.relative_to(ROOT)),
    ]
    run_command(record_command, 120, f"record quality {candidate}", record=False)
    refresh_report()
    commit(f"benchmark: record {candidate} quality smoke")
    return next((item for item in registry().get("quality", []) if item.get("candidate") == candidate), None)


def ensure_stage0() -> bool:
    if not ENVIRONMENT.exists():
        run_command(["uv", "run", "python", "qwen38-overnight/harness/collect_environment.py"], 120, "collect environment")
    if not (OUT / "harness" / "watchdog-test.json").exists():
        run_command(
            [
                "uv", "run", "python", "qwen38-overnight/harness/watchdog.py",
                "--label", "watchdog-harmless-test",
                "--memory-log", "qwen38-overnight/harness/watchdog-test-memory.log",
                "--child-log", "qwen38-overnight/harness/watchdog-test-child.log",
                "--result-json", "qwen38-overnight/harness/watchdog-test.json",
                "--child-result", "qwen38-overnight/harness/watchdog-test-child.json",
                "--progress-file", "qwen38-overnight/harness/watchdog-test.progress",
                "--timeout", "30", "--", "/bin/sleep", "3",
            ],
            60,
            "test harmless watchdog",
        )
    value = subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    return value == "0"


def main() -> int:
    if not ensure_stage0():
        print("Safety gate failed or was unreadable; no model stages will run.", file=sys.stderr)
        return 2
    refresh_report()

    # Stage 1: uniform 3-bit text-only target and conditional native MTP/APC.
    candidate = "q38-3bit"
    models = MODELS[candidate]
    tiny = model_stage(candidate, "target-tiny", models["target"], "target", "tiny", 480, 12)
    # Preserve each adapter failure, then advance through the installed
    # mlx-lm text-backend bridges. Re-read the latest failure after every
    # existing stage so a prior run cannot strand the state machine.
    retry_specs = (
        ("target-tiny-text-only", ("missing", "vision")),
        ("target-tiny-text-only-embedding", ("token embeddings",)),
        ("target-tiny-text-only-stopping", ("stopping_criteria",)),
        ("target-tiny-text-only-detokenizer", ("skip_special_token_ids",)),
    )
    for retry_stage, fragments in retry_specs:
        if healthy(tiny):
            break
        error_text = str((tiny or {}).get("result", {}).get("error", "")).lower()
        if all(fragment in error_text for fragment in fragments):
            tiny = model_stage(
                candidate,
                retry_stage,
                models["target"],
                "target",
                "tiny",
                480,
                12,
            )
    if healthy(tiny):
        target = model_stage(candidate, "target-practical", models["target"], "target", "practical", 1200)
    else:
        target = None
    if healthy(target):
        mtp_tiny = model_stage(candidate, "mtp-tiny", models["target"], "mtp", "tiny", 600, 12)
        mtp = model_stage(candidate, "mtp-practical", models["target"], "mtp", "practical", 1500) if healthy(mtp_tiny) else None
        target_rate = target.get("metrics", {}).get("decode_tok_s") if target else None
        mtp_rate = mtp.get("metrics", {}).get("decode_tok_s") if mtp else None
        multiplier = mtp_rate / target_rate if isinstance(target_rate, (int, float)) and isinstance(mtp_rate, (int, float)) and target_rate > 0 else None
        if healthy(mtp) and isinstance(multiplier, (int, float)) and multiplier >= 1.05:
            apc_stage(candidate, models["target"], "mtp", 8108)
        else:
            apc_stage(candidate, models["target"], "target", 8108)

    # Stage 3: mixed 3.5 bpw. The stock runtime is known from metadata to
    # instantiate the multimodal Qwen3.5 wrapper; only a guarded tiny run can
    # establish whether the working set is safe enough for a practical run.
    candidate = "q38-mixed-3.5"
    models = MODELS[candidate]
    tiny = model_stage(candidate, "target-tiny", models["target"], "target", "tiny", 600, 12)
    if healthy(tiny):
        target = model_stage(candidate, "target-practical", models["target"], "target", "practical", 1500)
    else:
        target = None
    if healthy(target, conservative=True):
        mtp_tiny = model_stage(candidate, "mtp-tiny", models["target"], "mtp", "tiny", 600, 12)
        if healthy(mtp_tiny, conservative=True):
            model_stage(candidate, "mtp-practical", models["target"], "mtp", "practical", 1500)
        # APC is deferred unless this target is clearly faster than the 3-bit
        # configuration; this avoids a completeness-only high-pressure test.
        mixed_rate = target.get("metrics", {}).get("decode_tok_s") if target else None
        base = existing_run("q38-3bit", "target-practical")
        base_rate = base.get("metrics", {}).get("decode_tok_s") if base else None
        if isinstance(mixed_rate, (int, float)) and isinstance(base_rate, (int, float)) and mixed_rate >= base_rate * 1.10:
            apc_stage(candidate, models["target"], "target", 8109)

    # Stage 4: one 2-bit target-only control/fallback. MTP/APC optimization is
    # postponed until the deterministic quality gate says it remains serious.
    candidate = "q38-2bit"
    models = MODELS[candidate]
    tiny = model_stage(candidate, "target-tiny", models["target"], "target", "tiny", 480, 12)
    if not healthy(tiny) and "TokenizerWrapper" in str((tiny or {}).get("result", {}).get("error", "")):
        tiny = model_stage(
            candidate,
            "target-tiny-text-only-fixed",
            models["target"],
            "target",
            "tiny",
            480,
            12,
        )
    target = model_stage(candidate, "target-practical", models["target"], "target", "practical", 1200) if healthy(tiny) else None

    # Stage 5: quality smoke for every candidate that completed a practical run.
    for candidate in ("q38-3bit", "q38-mixed-3.5", "q38-2bit"):
        run = existing_run(candidate, "target-practical")
        if healthy(run):
            quality_stage(candidate, MODELS[candidate]["target"])

    # Only optimize 2-bit after its quality gate, and only when it is either a
    # fallback or surprisingly coherent. No block-size or quantization sweep.
    q2 = next((item for item in registry().get("quality", []) if item.get("candidate") == "q38-2bit"), None)
    three = existing_run("q38-3bit", "target-practical")
    if healthy(existing_run("q38-2bit", "target-practical")) and isinstance(q2, dict) and q2.get("passed", 0) >= 3 and not healthy(three):
        models = MODELS["q38-2bit"]
        mtp_tiny = model_stage("q38-2bit", "mtp-tiny", models["target"], "mtp", "tiny", 600, 12)
        if healthy(mtp_tiny):
            model_stage("q38-2bit", "mtp-practical", models["target"], "mtp", "practical", 1500)
            mtp = existing_run("q38-2bit", "mtp-practical")
            if healthy(mtp):
                apc_stage("q38-2bit", models["target"], "mtp", 8110)

    refresh_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
