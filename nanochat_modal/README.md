# nanochat SFT on Modal

Supervised Fine-Tuning (SFT) pipeline running on [Modal](https://modal.com) cloud GPUs. Trains [karpathy/nanochat](https://github.com/karpathy/nanochat) models (d6 = 28M params) on a SmolTalk + MMLU + GSM8K mixture.

## Prerequisites

- A [Modal](https://modal.com) account with billing enabled
- Modal CLI installed and configured:

```bash
pip install modal
modal setup
```

- A Modal Volume named `nanochat-vol` containing:
  - Pretrain base checkpoint at `/base_checkpoints/d6/`
  - Tokenizer files at `/tokenizer/`

## Quick Start

```bash
cd training/

# Run SFT synchronously (live logs, Ctrl+C to cancel)
modal run nanochat_sft.py::run_sft

# Run detached (continues after terminal closes)
modal run --detach nanochat_sft.py

# Test inference on the trained checkpoint
modal run nanochat_sft.py::test_inference --prompt "Hello!"

# Download latest checkpoint from the volume
modal run nanochat_sft.py::download_sft

# Run ChatCORE evaluation on 2×A10G
modal run nanochat_sft.py::run_eval
```

## Using the Shell Wrapper

The included `run_sft.sh` script provides convenient shortcuts for all common operations:

```bash
./run_sft.sh run          # synchronous SFT (live logs)
./run_sft.sh detach       # detached SFT (background)
./run_sft.sh logs         # tail logs from latest/active run
./run_sft.sh infer "Hi!"  # test inference
./run_sft.sh status       # inspect volume contents
./run_sft.sh download     # download checkpoint from volume
./run_sft.sh eval         # run ChatCORE evaluation
```

## Architecture

| Component | Detail |
|---|---|
| Modal App | `nanochat-sft` |
| Container | debian-slim + Python 3.11 + PyTorch 2.3.0 |
| GPU | 2× A10G (fallback supports A100/H100) |
| Volume mount | `/vol` → `nanochat-vol` |
| Checkpoint path | `/vol/chatsft_checkpoints/d6/` |

The pipeline:

1. **`setup_nanochat()`** — Clones the nanochat repo and applies compatibility patches.
2. **`run_sft()`** — Downloads the pretrain base checkpoint, trains the tokenizer, and runs `torchrun` for 1500 steps on 2×A10G.
3. **`test_inference()`** — Quick inference smoke test on a single T4 GPU.
4. **`download_sft()`** — Copies the checkpoint from the Modal volume to the local container.
5. **`run_eval()`** — Runs ChatCORE benchmark evaluation on 2×A10G.

## Patches Applied

The `setup_nanochat()` function applies these patches to the cloned nanochat repo for compatibility:

1. **`dataset.py`** — `base_data_climbmix` → `base_data` (dataset path fix)
2. **`engine.py`** — dtype configurable via `NANOCHAT_DTYPE` env var
3. **`gpt.py`** — `F.rms_norm` → manual RMSNorm (PyTorch 2.3 compatibility)
4. **`chat_sft.py`** — `torch._dynamo.config.suppress_errors = True`
5. **`flash_attention.py`** — Strip `enable_gqa` kwarg (PyTorch < 2.5 compatibility)

## Results

| Metric | Value |
|---|---|
| Steps trained | 971 (1 epoch) |
| Final val_bpb | 0.4891 |
| Model | d6 (6 layers, 384 dim, 28M params) |
| GPU time | ~35 min on 2× A10G |
| Cost | ~$1.20 |

## Directory Structure

```
nanochat_modal/
  nanochat/                    # Upstream karpathy/nanochat repo (embedded git — do not modify)
  training/                    # Modal training scripts
    nanochat_sft.py            # Main Modal app
    run_sft.sh                 # Shell wrapper for common operations
  checkpoints/
    pretrain/                  # Base pretrain checkpoint (step 8600, d6)
      model_008600.pt          #   Pretrain model weights (281 MB)
      meta_*.json              #   Training metadata (steps 2000-8600)
      optim_008600_rank*.pt    #   Optimizer states (2 × 261 MB)
    sft/                       # SFT checkpoints downloaded from Modal volume
      model_000971.pt          #   SFT model weights (281 MB)
      meta_000971.json         #   Training metadata
  nanochat_sft_modal_runbook.* # Runbook documentation (md + docx)
  README.md
```

## Troubleshootingnanochat_sft_modal_runbook.md](nanochat_sft_modal_runbook.md) — Full runbook with setup instructions, troubleshooting, and batch size math
- [nanochat_sft_modal_runbook.docx](nanochat_sft_modal_runbook.docx) — Same runbook in Word format

## Troubleshooting

- **NaN loss during training** — Pass `--load-optimizer 0` to avoid inheriting optimizer state from the pretrain checkpoint.
- **`enable_gqa` error** — PyTorch < 2.5 needs the `flash_attention.py` patch (applied automatically by `setup_nanochat()`).
- **`psutil` error** — Missing package in the container image (should be fixed in the current script; if it recurs, add `psutil` to the image dependencies).
- **ZERO assistant tokens in predictions** — Use `--max-seq-len 2048`; conversations are typically 800–1400 tokens and may overflow shorter limits.

## License

MIT — same as [nanochat](https://github.com/karpathy/nanochat).
