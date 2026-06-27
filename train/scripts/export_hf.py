"""Merge LoRA adapter and export HF-ready weights to /vol/exports/."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_config, resolve_adapter_path, resolve_model_path
from volumes import DEFAULT_MODEL_ID, EXPORTS_DIR, ensure_volume_layout


def export_merged(cfg: dict[str, Any], *, volume: Any | None = None) -> Path:
    ensure_volume_layout()
    model_id = cfg.get("model_id", DEFAULT_MODEL_ID)
    project = cfg.get("project_name", "mars")
    export_name = cfg.get("export_name", f"{project}-merged")
    out_dir = EXPORTS_DIR / export_name

    adapter_path = resolve_adapter_path(cfg, default_stage=cfg.get("source_stage", "grpo"))
    if not adapter_path:
        raise FileNotFoundError("No adapter found to export — set adapter_path or previous_stage")

    base_path = resolve_model_path(model_id, cfg)
    print(f"Merging adapter {adapter_path} into {base_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged = model.merge_and_unload()

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))

    meta = {
        "base_model": model_id,
        "adapter_path": str(adapter_path),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "project_name": project,
        "vllm_serve": f"vllm serve {out_dir} --enable-auto-tool-choice --tool-call-parser lfm2",
        "sampling": {"temperature": 0.2, "top_k": 80, "repetition_penalty": 1.05},
    }
    (out_dir / "mars_export_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Exported merged model → {out_dir}")

    hf_repo = cfg.get("hf_repo")
    if hf_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(hf_repo, exist_ok=True, repo_type="model")
        api.upload_folder(folder_path=str(out_dir), repo_id=hf_repo, repo_type="model")
        print(f"Pushed to HuggingFace: {hf_repo}")

    if volume is not None:
        volume.commit()
    return out_dir


def main(config_path: str) -> None:
    export_merged(load_config(config_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
