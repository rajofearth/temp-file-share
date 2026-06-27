"""TRL GRPOTrainer with verifiable rewards for agent harness alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trl import GRPOConfig, GRPOTrainer

from common import (
    VolumeCommitCallback,
    load_config,
    load_model_and_tokenizer,
    prepare_run,
    save_final,
    write_run_meta,
)
from dataset_loaders import build_grpo_dataset
from rewards import REWARD_REGISTRY
from volumes import DEFAULT_MODEL_ID


def _build_reward_func(cfg: dict[str, Any]):
    specs = cfg.get("rewards", {})
    names = specs.get("funcs", ["lfm_tool_format", "tool_json_args", "concise", "solution_match"])
    weights = specs.get("weights")

    funcs = [REWARD_REGISTRY[n] for n in names]

    if not weights:
        return funcs

    def combined(completions, **kwargs):  # noqa: ANN001
        totals = [0.0] * len(completions)
        for fn, weight in zip(funcs, weights):
            scores = fn(completions=completions, **kwargs)
            totals = [t + weight * s for t, s in zip(totals, scores)]
        return totals

    return combined


def run_grpo(cfg: dict[str, Any], *, volume: Any | None = None) -> Path:
    output_dir, resume_from, adapter_path = prepare_run(cfg, "grpo")
    model_id = cfg.get("model_id", DEFAULT_MODEL_ID)
    training = cfg.get("training", {})
    peft_cfg = cfg.get("peft", {})

    if not adapter_path and not cfg.get("previous_stage"):
        raise ValueError(
            "GRPO requires `previous_stage` or `adapter_path` pointing to DPO/SFT LoRA adapter"
        )

    if resume_from:
        print(f"Resuming GRPO from {resume_from}")
    else:
        write_run_meta(
            output_dir,
            cfg,
            stage="grpo",
            adapter_path=str(adapter_path) if adapter_path else None,
        )

    print("Building GRPO dataset...")
    train_dataset = build_grpo_dataset(cfg["dataset"])
    print(f"Training on {len(train_dataset)} prompts")

    model, tokenizer, lora_config = load_model_and_tokenizer(
        model_id,
        cfg,
        adapter_path=adapter_path,
        peft_cfg=peft_cfg if not adapter_path else None,
    )

    reward_func = _build_reward_func(cfg)
    save_steps = int(training.get("save_steps", 100))

    # A10G: use_vllm=False (MoE 8B + colocated vLLM OOMs). A100: set use_vllm=true in config.
    use_vllm = bool(training.get("use_vllm", False))

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(training.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 8)),
        learning_rate=float(training.get("learning_rate", 1e-6)),
        lr_scheduler_type=training.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        logging_steps=int(training.get("logging_steps", 5)),
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=int(training.get("save_total_limit", 3)),
        bf16=True,
        gradient_checkpointing=True,
        num_generations=int(training.get("num_generations", 4)),
        max_completion_length=int(training.get("max_completion_length", 512)),
        temperature=float(training.get("temperature", 0.7)),
        beta=float(training.get("beta", 0.0)),
        use_vllm=use_vllm,
        vllm_mode=training.get("vllm_mode", "colocate"),
        vllm_gpu_memory_utilization=float(training.get("vllm_gpu_memory_utilization", 0.25)),
        report_to=training.get("report_to", "none"),
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        peft_config=lora_config,
        callbacks=[VolumeCommitCallback(volume)],
    )

    trainer.train(resume_from_checkpoint=resume_from)
    final_dir = save_final(trainer, output_dir, tokenizer, volume)
    print(f"GRPO complete → {final_dir}")
    return output_dir


def main(config_path: str) -> None:
    run_grpo(load_config(config_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
