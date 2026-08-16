"""Collect reproducibility and baseline safety information."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from memory_utils import snapshot


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "qwen38-overnight"
PROMPT_SOURCE = ROOT / "coding_prompt.md"
PROMPT_DEST = OUT / "coding-prompt.txt"


def command(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    output = (result.stdout or result.stderr).strip()
    return output if output else f"exit={result.returncode}"


def redacted_hardware_report() -> str:
    raw = command(["system_profiler", "SPHardwareDataType"])
    private_labels = (
        "Serial Number",
        "Hardware UUID",
        "Provisioning UDID",
        "Activation Lock Status",
    )
    return "\n".join(
        line for line in raw.splitlines() if not any(label in line for label in private_labels)
    )


def package_versions() -> dict[str, str]:
    names = [
        "mlx",
        "mlx-lm",
        "mlx-vlm",
        "mlx-dspark",
        "transformers",
        "huggingface-hub",
        "httpx",
    ]
    values: dict[str, str] = {}
    for name in names:
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = "not-installed"
    return values


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    source_bytes = PROMPT_SOURCE.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if PROMPT_DEST.exists() and hashlib.sha256(PROMPT_DEST.read_bytes()).hexdigest() != source_hash:
        raise SystemExit(f"refusing to overwrite differing prompt: {PROMPT_DEST}")
    if not PROMPT_DEST.exists():
        shutil.copyfile(PROMPT_SOURCE, PROMPT_DEST)

    baseline_samples = [snapshot()]
    time.sleep(1.0)
    baseline_samples.append(snapshot())
    time.sleep(1.0)
    baseline_samples.append(snapshot())
    info: dict[str, Any] = {
        "machine": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "chip_and_memory": redacted_hardware_report(),
            "physical_memory_bytes": command(["sysctl", "-n", "hw.memsize"]),
            "macos": command(["sw_vers"]),
            "disk_free": command(["df", "-h", "."]),
        },
        "safety": {
            "iogpu.wired_limit_mb": command(["sysctl", "-n", "iogpu.wired_limit_mb"]),
            "policy_confirmed_zero": command(["sysctl", "-n", "iogpu.wired_limit_mb"]) == "0",
        },
        "tooling": {
            "uv": command(["uv", "--version"]),
            "python": sys.version,
            "python_version": platform.python_version(),
            "packages": package_versions(),
        },
        "prompt": {
            "source": str(PROMPT_SOURCE),
            "copied_to": str(PROMPT_DEST),
            "sha256": source_hash,
            "bytes": len(source_bytes),
            "characters": len(source_bytes.decode("utf-8")),
        },
        "baseline_memory_samples": baseline_samples,
    }
    (OUT / "environment.json").write_text(
        json.dumps(info, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Qwen3.8 overnight environment",
        "",
        "Safety gate: `iogpu.wired_limit_mb` must remain zero for all runs.",
        "",
        "## Machine",
        "",
        f"- Platform: `{info['machine']['platform']}`",
        f"- Architecture: `{info['machine']['architecture']}`",
        f"- Physical memory: `{info['machine']['physical_memory_bytes']}` bytes",
        "- Hardware report:",
        "",
        "```text",
        info["machine"]["chip_and_memory"],
        "```",
        "",
        "- macOS:",
        "",
        "```text",
        info["machine"]["macos"],
        "```",
        "",
        f"- Free disk at start: `{info['machine']['disk_free']}`",
        f"- `iogpu.wired_limit_mb`: `{info['safety']['iogpu.wired_limit_mb']}`",
        "",
        "## Tooling",
        "",
        f"- uv: `{info['tooling']['uv']}`",
        f"- Python: `{info['tooling']['python_version']}`",
    ]
    for name, version in info["tooling"]["packages"].items():
        lines.append(f"- {name}: `{version}`")
    lines += [
        "",
        "## Prompt provenance",
        "",
        f"- Original: `{PROMPT_SOURCE}`",
        f"- Copied prompt: `{PROMPT_DEST}`",
        f"- SHA256: `{source_hash}`",
        f"- Size: {len(source_bytes)} bytes / {len(source_bytes.decode('utf-8'))} UTF-8 characters",
        "- Actual model-formatted token counts are recorded in each model run JSON.",
        "",
        "## Baseline idle memory",
        "",
        "The same `memory_pressure -Q`, `vm_stat`, and `sysctl vm.swapusage` metrics used by the watchdog were sampled three times before model work.",
        "",
        "```json",
        json.dumps(baseline_samples, indent=2, sort_keys=True),
        "```",
        "",
    ]
    (OUT / "environment.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(info, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
