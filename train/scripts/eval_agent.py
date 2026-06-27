"""Agent eval harness — tool format, qualitative suite, regression vs stock LFM."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_config, resolve_adapter_path, resolve_model_path
from dataset_mix import TOOL_CALL_END, TOOL_CALL_START
from rewards import lfm_tool_format_reward, tool_json_args_reward
from volumes import DEFAULT_MODEL_ID, EVALS_DIR, ensure_volume_layout

# 12-question qualitative suite (adapted from archive/nanochat_modal/docs/COMPARISON.md)
DEFAULT_QUESTIONS = [
    "What is the capital of France?",
    "Explain why the sky is blue in one paragraph.",
    "Write a short poem about artificial intelligence.",
    "What is 15 × 37?",
    "Explain the difference between a CPU and a GPU in one sentence.",
    "Write a Python function that checks if a string is a palindrome.",
    "How many 'r' are in the word 'strawberry'?",
    "What is the tallest mountain in the world?",
    "Explain quantum computing in simple terms.",
    "What are the three branches of the US government?",
    "Write a haiku about the ocean.",
    "Respond with exactly one sentence: why is open-source software valuable?",
]

TOOL_EVAL_CASES = [
    {
        "tools": '[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}]',
        "query": "What's the weather in Boston?",
        "expect_tool": "get_weather",
    },
    {
        "tools": '[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]',
        "query": "Read the contents of config.yaml",
        "expect_tool": "read_file",
    },
]


def _load_eval_model(cfg: dict[str, Any]):
    model_id = cfg.get("model_id", DEFAULT_MODEL_ID)
    adapter_path = resolve_adapter_path(cfg) or resolve_adapter_path(
        {"previous_stage": cfg.get("previous_stage"), "project_name": cfg.get("project_name", "")},
        default_stage=cfg.get("stage", "grpo"),
    )
    model_path = resolve_model_path(model_id, cfg)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def _generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 512) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_k=80,
            repetition_penalty=1.05,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=False)


def eval_tool_format(completions: list[str]) -> dict[str, float]:
    fmt = lfm_tool_format_reward(completions)
    args = tool_json_args_reward(completions)
    return {
        "lfm_tool_format_mean": sum(fmt) / len(fmt) if fmt else 0.0,
        "tool_json_args_mean": sum(args) / len(args) if args else 0.0,
        "samples": len(completions),
    }


def run_eval(cfg: dict[str, Any], *, volume: Any | None = None) -> Path:
    ensure_volume_layout()
    stage = cfg.get("stage_label", "eval")
    project = cfg.get("project_name", "mars")
    out_path = EVALS_DIR / f"{stage}_{project}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"

    print("Loading model for eval...")
    model, tokenizer = _load_eval_model(cfg)

    results: dict[str, Any] = {
        "stage": stage,
        "project_name": project,
        "model_id": cfg.get("model_id", DEFAULT_MODEL_ID),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_eval": [],
        "qualitative": [],
    }

    # Tool-call format eval (BFCL-lite)
    tool_completions = []
    for case in TOOL_EVAL_CASES:
        messages = [
            {"role": "system", "content": f"List of tools: {case['tools']}"},
            {"role": "user", "content": case["query"]},
        ]
        response = _generate(model, tokenizer, messages, max_new_tokens=256)
        tool_completions.append(response)
        hit = case["expect_tool"] in response and TOOL_CALL_START in response
        results["tool_eval"].append(
            {"query": case["query"], "expect_tool": case["expect_tool"], "hit": hit, "response": response[:500]}
        )

    results["tool_format_scores"] = eval_tool_format(tool_completions)
    results["tool_eval_accuracy"] = sum(1 for t in results["tool_eval"] if t["hit"]) / len(TOOL_EVAL_CASES)

    # Qualitative 12-question suite
    for q in cfg.get("questions", DEFAULT_QUESTIONS):
        messages = [{"role": "user", "content": q}]
        response = _generate(model, tokenizer, messages)
        results["qualitative"].append({"question": q, "response": response[:800]})

    out_path.write_text(json.dumps(results, indent=2))
    print(f"Eval saved → {out_path}")

    if volume is not None:
        volume.commit()
    return out_path


def main(config_path: str) -> None:
    run_eval(load_config(config_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
