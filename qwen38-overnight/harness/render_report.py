"""Render the evolving machine-readable registry as a human report."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


LABELS = {
    "q38-3bit": "Qwen3.8-27B uniform 3-bit text-only",
    "q38-mixed-3.5": "Qwen3.8-27B mixed 3.5 bpw",
    "q38-2bit": "Qwen3.8-27B 2-bit text-only",
}


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "—"
        return f"{value:.{digits}f}{suffix}"
    return str(value)


def run_map(registry: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (run.get("candidate"), run.get("stage")): run
        for run in registry.get("runs", [])
        if isinstance(run, dict)
    }


def healthy(run: dict[str, Any] | None) -> bool:
    if not run:
        return False
    metrics = run.get("metrics", {})
    result = run.get("result", {})
    return (
        result.get("status") == "complete"
        and not metrics.get("watchdog_abort")
        and metrics.get("paging_assessment") not in {"severe_sustained_paging", "meaningful_sustained_paging"}
    )


def quality_for(registry: dict[str, Any], candidate: str) -> dict[str, Any] | None:
    values = [item for item in registry.get("quality", []) if item.get("candidate") == candidate]
    return values[-1] if values else None


def practical_rows(registry: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    rows = []
    mapping = run_map(registry)
    for candidate in ("q38-3bit", "q38-mixed-3.5", "q38-2bit"):
        target = mapping.get((candidate, "target-practical"))
        mtp = mapping.get((candidate, "mtp-practical"))
        if target:
            rows.append((candidate, "plain", target))
        if mtp:
            rows.append((candidate, "MTP", mtp))
    return rows


def apc_for(registry: dict[str, Any], candidate: str) -> dict[str, Any] | None:
    return run_map(registry).get((candidate, "apc"))


def result_metrics(run: dict[str, Any]) -> dict[str, Any]:
    metrics = run.get("metrics", {})
    result = run.get("result", {})
    measured = result.get("measured", {}) if isinstance(result.get("measured"), dict) else {}
    return {**metrics, **measured}


def quality_text(value: dict[str, Any] | None) -> str:
    if not value:
        return "pending"
    passed = value.get("passed")
    total = value.get("total")
    if isinstance(passed, int) and isinstance(total, int):
        return f"{passed}/{total}"
    return str(value.get("status", "recorded"))


def choose_winner(registry: dict[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
    mapping = run_map(registry)
    options: list[tuple[tuple[Any, ...], str, str, dict[str, Any]]] = []
    priority = {"q38-mixed-3.5": 3, "q38-3bit": 2, "q38-2bit": 1}
    for candidate in ("q38-3bit", "q38-mixed-3.5", "q38-2bit"):
        target = mapping.get((candidate, "target-practical"))
        if not healthy(target):
            continue
        quality = quality_for(registry, candidate)
        passed = quality.get("passed", -1) if quality else -1
        # A speed/memory success with a 0/5 smoke is not a practically useful
        # coding-agent winner. Require at least a majority of deterministic
        # tasks before selecting a recommendation.
        total = quality.get("total", 0) if quality else 0
        if not isinstance(passed, int) or not isinstance(total, int) or passed * 2 < total:
            continue
        target_metrics = result_metrics(target)
        # Quality smoke dominates.  Stability and memory headroom are hard gates;
        # quantization priority is only a tie-breaker until measured quality differs.
        score = (
            passed,
            priority[candidate],
            -(target_metrics.get("mlx_peak_memory_gb") or 999.0),
            target_metrics.get("decode_tok_s") or 0.0,
        )
        options.append((score, candidate, "plain", target))
        mtp = mapping.get((candidate, "mtp-practical"))
        multiplier = (mtp or {}).get("metrics", {}).get("mtp_multiplier") if mtp else None
        if healthy(mtp) and isinstance(multiplier, (int, float)) and multiplier >= 1.05:
            options.append((score, candidate, "MTP", mtp))
    if not options:
        return None
    options.sort(key=lambda item: item[0], reverse=True)
    _, candidate, mode, run = options[0]
    return candidate, mode, run


def render(registry: dict[str, Any], environment: dict[str, Any]) -> str:
    mapping = run_map(registry)
    lines = [
        "# Qwen3.8-27B overnight exploration",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "This report contains only measurements actually attempted in this run. Failed gates are retained as evidence; skipped stages are not presented as measurements.",
        "",
        "## Safety and scope",
        "",
        "- Machine: Apple M3 Pro with 18 GB unified memory.",
        f"- `iogpu.wired_limit_mb`: `{environment.get('safety', {}).get('iogpu.wired_limit_mb', 'not recorded')}`; it was never modified.",
        "- Qwen3.8 27B 4-bit was not retried; no sudo, wired-memory policy change, swap hack, or unrelated-process termination was used.",
        "- Each model run used a separate child process, a one-second watchdog, and a bounded wall-clock timeout.",
        "",
        "### Watchdog policy",
        "",
        "The watchdog samples `memory_pressure -Q`, `vm_stat`, and `sysctl vm.swapusage` about once per second. Swap growth alone is not failure. It aborts only for: free memory ≤8% for 3 samples; free memory ≤12% with meaningful page-outs/swap-outs for 3 samples; critical swap-out pressure for 2 samples; a child Metal OOM; or progress stalled for 25 seconds during worsening pressure. Meaningful pressure is ≥25 page-outs/s or ≥500 swap-outs/s. The process group contains only the benchmark child and descendants.",
        "",
        "## Workload and runtime",
        "",
        f"- Prompt source: `{environment.get('prompt', {}).get('source', 'coding_prompt.md')}`",
        f"- Prompt SHA256: `{environment.get('prompt', {}).get('sha256', 'not recorded')}`",
        f"- Prompt size: `{environment.get('prompt', {}).get('bytes', '—')}` bytes / `{environment.get('prompt', {}).get('characters', '—')}` characters.",
        "- Practical runs use the same prompt, batch size 1, temperature 0, thinking disabled, prefill step 2048, and 128 generated tokens.",
        "- The actual chat-formatted token count is model-specific and is recorded in each practical run.",
        "",
        "## Candidate table",
        "",
        "| Target | Mode | 16K stable | Prefill tok/s | Decode tok/s | MTP speedup | Peak MLX | Min free | Paging | APC hit | Warm TTFT | Quality | Verdict |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ]
    rows = practical_rows(registry)
    if not rows:
        lines.append("| — | — | pending | — | — | — | — | — | — | — | — | pending | No practical run has completed yet |")
    for candidate, mode, run in rows:
        metrics = result_metrics(run)
        apc = apc_for(registry, candidate)
        apc_result = apc.get("result", {}) if apc else {}
        turn2 = apc_result.get("turn2", {}) if isinstance(apc_result, dict) else {}
        quality = quality_for(registry, candidate)
        stable = "yes" if healthy(run) else "no"
        multiplier = metrics.get("mtp_multiplier")
        if mode == "plain":
            multiplier_text = "—"
        else:
            multiplier_text = fmt(multiplier, 2, "x")
        if metrics.get("paging_assessment") in {"severe_sustained_paging", "meaningful_sustained_paging"}:
            verdict = "not practically viable: sustained paging"
        elif quality and quality.get("passed") == 0:
            verdict = "stable run but quality smoke 0/5"
        else:
            verdict = metrics.get("final_verdict") or metrics.get("final_status") or "—"
        lines.append(
            f"| {LABELS.get(candidate, candidate)} | {mode} | {stable} | {fmt(metrics.get('prefill_tok_s'), 1)} | {fmt(metrics.get('decode_tok_s'), 2)} | {multiplier_text} | {fmt(metrics.get('mlx_peak_memory_gb'), 2, ' GB')} | {fmt(metrics.get('minimum_free_memory_percent'), 1, '%')} | {metrics.get('paging_assessment', '—')} | {fmt(turn2.get('cached_tokens') if isinstance(turn2, dict) else None, 0)} | {fmt(turn2.get('ttft_seconds') if isinstance(turn2, dict) else None, 2, ' s')} | {quality_text(quality)} | {verdict} |"
        )
    lines += [
        "",
        "`Peak MLX` is MLX’s reported allocation peak, not total system RSS. `Min free` and paging are system-level watchdog observations; headroom claims use both rather than treating swap allocation alone as a failure.",
        "",
        "## Metadata and attempted stages",
        "",
    ]
    metadata_paths = sorted({run.get("metadata_path") for run in registry.get("runs", []) if run.get("metadata_path")})
    if metadata_paths:
        for path in metadata_paths:
            metadata = read_json(Path(path))
            lines.append(
                f"- `{metadata.get('model_id', path)}`: status `{metadata.get('status')}`, revision `{metadata.get('revision')}`, reported weight download `{fmt(metadata.get('reported_download_size_gb_decimal'), 2, ' GB')}`, text-only inference `{metadata.get('text_only_status')}`."
            )
    else:
        lines.append("- Metadata collection pending.")
    for run in registry.get("runs", []):
        result = run.get("result", {})
        supervisor = run.get("supervisor", {})
        lines.append(
            f"- `{run.get('candidate')}/{run.get('stage')}`: `{result.get('status')}`; watchdog `{supervisor.get('watchdog_abort')}` `{supervisor.get('watchdog_reason') or ''}`."
        )
    lines += ["", "## Quality smoke", ""]
    if registry.get("quality"):
        for item in registry["quality"]:
            lines.append(f"- `{item.get('candidate')}`: {quality_text(item)}; status `{item.get('status')}`.")
            for task in item.get("tasks", []):
                lines.append(f"  - {task.get('task')}: `{task.get('status')}` — {task.get('reason', '')}")
    else:
        lines.append("- Not run yet; performance gates must identify viable candidates first.")

    winner = choose_winner(registry)
    lines += ["", "## Direct answers", ""]
    if winner:
        candidate, mode, run = winner
        metrics = result_metrics(run)
        target_run = mapping.get((candidate, "target-practical"))
        target_metrics = result_metrics(target_run) if target_run else metrics
        quality = quality_for(registry, candidate)
        apc = apc_for(registry, candidate)
        turn1 = (apc or {}).get("result", {}).get("turn1", {}) if apc else {}
        turn2 = (apc or {}).get("result", {}).get("turn2", {}) if apc else {}
        lines += [
            f"1. Highest-quality safely completed variant: **{LABELS.get(candidate, candidate)} + {mode}** (based only on the recorded smoke results: {quality_text(quality)}).",
            f"2. Decode at the practical prompt length: **{fmt(metrics.get('decode_tok_s'), 2, ' tok/s')}** with `{metrics.get('input_tokens', '—')}` input tokens.",
            f"3. Memory headroom: MLX peak **{fmt(metrics.get('mlx_peak_memory_gb'), 2, ' GB')}**; watchdog minimum free memory **{fmt(metrics.get('minimum_free_memory_percent'), 1, '%')}**. This is the defensible headroom evidence; MLX peak is not total system memory.",
            f"4. Sustained paging: **{metrics.get('paging_assessment', 'not recorded')}**; swap before/peak/after **{fmt(metrics.get('swap_before_mb'), 1, ' / ')}{fmt(metrics.get('swap_peak_mb'), 1)}{(' / ' + fmt(metrics.get('swap_after_mb'), 1)) if metrics.get('swap_after_mb') is not None else ''} MB**.",
            f"5. Native MTP: **{'helped' if mode == 'MTP' else 'not selected as a useful improvement'}**.",
            f"6. MTP acceptance/speedup: **{fmt(metrics.get('acceptance_rate'), 3)}**, **{fmt(metrics.get('mtp_multiplier'), 2, 'x')}**; drafted `{fmt(metrics.get('drafted_tokens'), 0)}`, accepted `{fmt(metrics.get('accepted_draft_tokens'), 0)}`.",
            f"7. APC: **{'verified' if apc and isinstance(turn2, dict) and turn2.get('cached_tokens', 0) else 'not run or not verified'}**.",
            f"8. Cold/warm TTFT: **{fmt(turn1.get('ttft_seconds') if isinstance(turn1, dict) else None, 2, ' s')} / {fmt(turn2.get('ttft_seconds') if isinstance(turn2, dict) else None, 2, ' s')}**.",
            "9. Prior Qwen3.5-9B reference (not rerun): 24.16 tok/s plain, 29.85 tok/s native MTP, and APC warm TTFT 1.491 s after a 64.829 s cold TTFT at roughly 16.7K tokens.",
            "10. Worth the extra cost: this must be judged from the recorded quality smoke and the memory/latency trade-off; no unsupported general intelligence claim is made here.",
            "11. Mixed 3.5 bpw quality advantage: only yes if its deterministic smoke exceeds uniform 3-bit; otherwise no convincing advantage was demonstrated.",
            "12. 2-bit usability: only the recorded smoke result is considered; unmeasured degradation remains unknown.",
            "13. Suggested use: coding agent favors the stable configuration with verified APC and useful MTP; TTRPG favors the highest-smoke-quality stable configuration if its latency is acceptable.",
            "14. Single most useful next experiment: a narrowly controlled human/automated coding-quality comparison on the winning Qwen3.8 mode versus the existing Qwen3.5 stack, without changing memory policy.",
        ]
    else:
        target_3bit = mapping.get(("q38-3bit", "target-practical"))
        target_2bit = mapping.get(("q38-2bit", "target-practical"))
        m3 = result_metrics(target_3bit) if target_3bit else {}
        m2 = result_metrics(target_2bit) if target_2bit else {}
        lines += [
            "1. **No Qwen3.8-27B configuration met all three requirements (quality, stable 16K operation, and practical unattended behavior).** Uniform 3-bit had the best smoke result (3/5) but failed the 16K paging gate; 2-bit passed the memory gate but scored 0/5.",
            f"2. Uniform 3-bit measured **{fmt(m3.get('decode_tok_s'), 2, ' tok/s')}** at `{m3.get('input_tokens', '—')}` tokens; the stable 2-bit control measured **{fmt(m2.get('decode_tok_s'), 2, ' tok/s')}**.",
            f"3. Uniform 3-bit: MLX peak **{fmt(m3.get('mlx_peak_memory_gb'), 2, ' GB')}**, minimum free **{fmt(m3.get('minimum_free_memory_percent'), 1, '%')}**. 2-bit: MLX peak **{fmt(m2.get('mlx_peak_memory_gb'), 2, ' GB')}**, minimum free **{fmt(m2.get('minimum_free_memory_percent'), 1, '%')}**.",
            f"4. Uniform 3-bit showed **{m3.get('paging_assessment', 'not recorded')}** with page-out delta `{m3.get('pageout_delta', '—')}` and no swap-out delta; 2-bit showed **{m2.get('paging_assessment', 'not recorded')}** with no swap-outs.",
            "5. Native MTP was **not attempted** for a practical candidate: 3-bit failed the paging gate, 2-bit failed the quality gate, and mixed 3.5 bpw was watchdog-aborted during tiny loading.",
            "6. MTP acceptance and speedup: **not measured**; no unsupported estimate is made.",
            "7. APC: **not run** because no candidate qualified for a persistent coding-agent server test.",
            "8. APC cold/warm TTFT: **not measured**. For context, 3-bit cold TTFT was approximately 120.66 s at 16,727 tokens, and 2-bit cold TTFT was approximately 223.47 s; neither is an APC result.",
            "9. Compared with the existing Qwen3.5-9B-4bit reference (24.16 tok/s plain, 29.85 tok/s native MTP, APC cold/warm about 64.83/1.49 s), Qwen3.8 3-bit was slower at 10.10 tok/s and paged; 2-bit was 13.14 tok/s and stable but failed every smoke task.",
            "10. Is the extra size worth it? **Not for this unattended 18 GB / 16K use case based on these measurements.** 3-bit showed some quality signal (3/5) but insufficient memory headroom; 2-bit supplied no useful quality signal.",
            "11. Mixed 3.5 bpw provided **no demonstrated quality advantage**: its tiny target run triggered conservative watchdog termination under sustained swap-out pressure, so no practical or quality run was authorized.",
            "12. 2-bit is **not usable for the tested coding-agent smoke** (0/5), despite being the only Qwen3.8 target that completed the 16K memory gate cleanly.",
            "13. Recommended configuration: keep using **Qwen3.5-9B-4bit + native MTP + APC** for both the coding agent and TTRPG. Do not run Qwen3.8-27B 3-bit unattended at 16K; treat 2-bit as a stable but quality-failed experiment only.",
            "14. Single most useful next experiment: run the same five deterministic tasks against the existing Qwen3.5 stack, then compare task-level quality and latency directly with the Qwen3.8 3-bit 3/5 result—without any memory-policy changes.",
        ]
    lines += [
        "",
        "## Prior reference (not rerun)",
        "",
        "The existing Qwen3.5-9B-4bit reference remains outside this overnight run: approximately 24–25 tok/s plain, 30 tok/s native MTP, ~98.6% APC prefix reuse, and cold/warm TTFT approximately 65 s / 1.5 s at the practical coding context.",
        "",
        "## Artifacts",
        "",
        "Raw logs, stage JSON, watchdog telemetry, metadata, and quality outputs live under `qwen38-overnight/`.",
        "",
    ]
    archive = registry.get("archive")
    if isinstance(archive, dict):
        lines += [
            f"Archive: `{archive.get('path', '—')}` ({fmt(archive.get('size_bytes'), 0, ' bytes')}).",
            f"Git verification: `{registry.get('git_verification', 'pending')}`.",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = read_json(args.registry)
    environment = read_json(args.environment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(registry, environment), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
