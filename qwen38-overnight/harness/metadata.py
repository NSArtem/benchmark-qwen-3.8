"""Fetch lightweight Hugging Face metadata/configuration without model weights."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


def json_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def config_for(repo_id: str, filename: str, revision: str | None) -> tuple[str | None, Any]:
    try:
        path = hf_hub_download(repo_id, filename=filename, revision=revision)
    except Exception:
        return None, None
    return path, json_load(Path(path))


def sibling_record(sibling: Any) -> dict[str, Any]:
    return {
        "rfilename": getattr(sibling, "rfilename", None),
        "size": getattr(sibling, "size", None),
        "blob_id": getattr(sibling, "blob_id", None),
        "lfs": getattr(sibling, "lfs", None),
    }


def infer_text_only(repo_id: str, config: Any, file_names: list[str]) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {"text_only": None, "basis": "config_unavailable"}
    lower_id = repo_id.lower()
    if "textonly" in lower_id or "text-only" in lower_id:
        return {"text_only": True, "basis": "repository_name"}
    vision_keys = {
        "vision_config",
        "vision_model",
        "visual",
        "image_token_id",
        "video_token_id",
        "image_start_token_id",
        "image_end_token_id",
    }
    present = sorted(key for key in vision_keys if key in config)
    processor_files = [name for name in file_names if "processor" in name.lower() or "preprocessor" in name.lower()]
    if present or processor_files:
        return {
            "text_only": False,
            "basis": "vision_or_processor_metadata_present",
            "vision_keys": present,
            "processor_files": processor_files,
        }
    return {"text_only": True, "basis": "no_vision_or_processor_metadata_in_inspected_files"}


def inspect_model(repo_id: str, revision: str | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "model_id": repo_id,
        "requested_revision": revision,
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "metadata_failed",
    }
    try:
        info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    except Exception as exc:
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
        return record
    siblings = [sibling_record(item) for item in (info.siblings or [])]
    file_names = [item["rfilename"] for item in siblings if item.get("rfilename")]
    weight_files = [
        item for item in siblings
        if isinstance(item.get("rfilename"), str)
        and item["rfilename"].lower().endswith((".safetensors", ".bin", ".npz", ".gguf"))
    ]
    sizes = [item["size"] for item in weight_files if isinstance(item.get("size"), int)]
    configs: dict[str, Any] = {}
    downloaded_config_paths: dict[str, str] = {}
    for filename in (
        "config.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "generation_config.json",
    ):
        if filename not in file_names:
            continue
        path, value = config_for(repo_id, filename, getattr(info, "sha", None) or revision)
        if path:
            downloaded_config_paths[filename] = path
        if value is not None:
            configs[filename] = value
    config = configs.get("config.json")
    quantization = None
    architecture = None
    context_metadata: dict[str, Any] = {}
    if isinstance(config, dict):
        quantization = config.get("quantization_config", config.get("quantization"))
        architecture = {
            "architectures": config.get("architectures"),
            "model_type": config.get("model_type"),
            "torch_dtype": config.get("torch_dtype"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
        }
        for key in (
            "max_position_embeddings",
            "max_seq_len",
            "max_sequence_length",
            "model_max_length",
            "max_context_length",
            "sliding_window",
        ):
            if key in config:
                context_metadata[key] = config[key]
    text_status = infer_text_only(repo_id, config, file_names)
    record.update(
        {
            "status": "ok",
            "revision": getattr(info, "sha", None),
            "last_modified": str(getattr(info, "lastModified", None)),
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "library_name": getattr(info, "library_name", None),
            "tags": getattr(info, "tags", None),
            "reported_download_size_bytes": sum(sizes) if sizes else None,
            "reported_download_size_gb_decimal": (sum(sizes) / 1e9 if sizes else None),
            "weight_file_count": len(weight_files),
            "weight_files": weight_files,
            "file_count": len(siblings),
            "files": siblings,
            "downloaded_config_paths": downloaded_config_paths,
            "configs": configs,
            "quantization_config": quantization,
            "architecture": architecture,
            "context_metadata": context_metadata,
            "text_only_status": text_status,
        }
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model in args.model:
        key = model.replace("/", "__")
        record = inspect_model(model, None)
        path = args.output_dir / f"{key}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"model": model, "status": record.get("status"), "path": str(path)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
