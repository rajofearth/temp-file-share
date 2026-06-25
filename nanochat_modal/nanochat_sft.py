"""Supervised Fine-Tuning of nanochat d6 on Modal.

Trains the d6 (6-layer, 384-dim, ~28M param) model on SmolTalk + MMLU + GSM8K
for 1500 steps. Uses 2x A10G GPUs for best cost/performance.

The pretrain checkpoint is cloned from GitHub in-container (fast), no local upload needed.

Prerequisites:
    pip install modal
    modal setup
    modal volume create nanochat-vol

Usage:
    # Full pipeline (checkpoint → tokenizer → SFT):
    modal run --detach nanochat_sft.py

    # Just SFT (assumes checkpoint + tokenizer already on volume):
    modal run --detach nanochat_sft.py::run_sft

    # Download the SFT checkpoint:
    modal run nanochat_sft.py::download_sft

    # Test inference with the SFT model:
    modal run nanochat_sft.py::test_inference --prompt "What is 2+2?"
"""

from pathlib import Path

import modal

# ── Paths ──────────────────────────────────────────────────────────────────

VOLUME_DIR = Path("/vol")
VOLUME_CHECKPOINT_DIR = VOLUME_DIR / "base_checkpoints" / "d6"
TOKENIZER_DIR = VOLUME_DIR / "tokenizer"
DATA_DIR = VOLUME_DIR / "base_data"
SFT_CHECKPOINT_DIR = VOLUME_DIR / "chatsft_checkpoints" / "d6"
IDENTITY_CONVOS = VOLUME_DIR / "identity_conversations.jsonl"

# ── Hyperparameters ────────────────────────────────────────────────────────

MAX_SEQ_LEN = 2048
DEVICE_BATCH_SIZE = 4
TOTAL_BATCH_SIZE = 524288
NUM_ITERATIONS = 1500
EVAL_EVERY = 200
EVAL_TOKENS = 131072

# ── Modal App & Volumes ────────────────────────────────────────────────────

app = modal.App("nanochat-sft")
volume = modal.Volume.from_name("nanochat-vol", create_if_missing=True)

# ── Container Image ────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "torchaudio==2.3.0",
        "rustbpe",
        "tiktoken",
        "datasets",
        "transformers",
        "httpx",
        "pyarrow",
        "sentencepiece",
        "wandb",
        "psutil",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)


# ── Helper: Setup nanochat repo ────────────────────────────────────────────


def setup_nanochat():
    """Clone nanochat if needed and apply patches."""
    import os
    import subprocess

    repo_path = "/nanochat"
    if not os.path.exists(repo_path):
        subprocess.run(
            ["git", "clone", "https://github.com/karpathy/nanochat.git", repo_path],
            check=True,
            capture_output=True,
        )
        print("✓ Cloned nanochat repo")

    os.chdir(repo_path)

    # Patch dataset.py: base_data_climbmix → base_data
    dset_path = os.path.join(repo_path, "nanochat", "dataset.py")
    with open(dset_path) as f:
        content = f.read()
    if "base_data_climbmix" in content:
        content = content.replace("base_data_climbmix", "base_data")
        with open(dset_path, "w") as f:
            f.write(content)
        print("✓ Patched dataset.py: climbmix → base_data")

    # Patch engine.py: make dtype configurable
    engine_path = os.path.join(repo_path, "nanochat", "engine.py")
    with open(engine_path) as f:
        content = f.read()
    old_line = 'dtype = torch.bfloat16 if device.type == "cuda" else torch.float32'
    new_line = (
        'dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, '
        '"float32": torch.float32}.get(os.environ.get("NANOCHAT_DTYPE", ""), '
        'torch.bfloat16 if device.type == "cuda" else torch.float32)'
    )
    if old_line in content:
        content = content.replace(old_line, new_line)
        with open(engine_path, "w") as f:
            f.write(content)
        print("✓ Patched engine.py: dtype respects NANOCHAT_DTYPE")
    elif "NANOCHAT_DTYPE" not in content:
        print("⚠ engine.py pattern mismatch")

    # Patch gpt.py: replace F.rms_norm (PyTorch 2.4+) with manual impl for 2.3.0 compat
    gpt_path = os.path.join(repo_path, "nanochat", "gpt.py")
    with open(gpt_path) as f:
        content = f.read()
    old_norm = "def norm(x):\n    return F.rms_norm(x, (x.size(-1),))"
    new_norm = (
        "def norm(x):\n"
        "    # Manual RMSNorm for PyTorch < 2.4 compatibility\n"
        "    var = x.pow(2).float().mean(-1, keepdim=True)\n"
        "    return (x.float() * torch.rsqrt(var + 1e-6)).to(x.dtype)"
    )
    if old_norm in content:
        content = content.replace(old_norm, new_norm)
        with open(gpt_path, "w") as f:
            f.write(content)
        print("✓ Patched gpt.py: F.rms_norm → manual RMSNorm")
    else:
        print("⚠ gpt.py norm function already patched or pattern changed")

    # Patch chat_sft.py: suppress torch.compile Dynamo errors
    sft_path = os.path.join(repo_path, "scripts", "chat_sft.py")
    with open(sft_path) as f:
        content = f.read()
    old_sft = "# ── Batch sizes"
    new_sft = (
        "import torch._dynamo\n"
        "torch._dynamo.config.suppress_errors = True  # fall back to eager on Dynamo errors\n"
        "# ── Batch sizes"
    )
    if "suppress_errors" not in content:
        content = content.replace(old_sft, new_sft)
        with open(sft_path, "w") as f:
            f.write(content)
        print("✓ Patched chat_sft.py: suppress_errors=True")
    else:
        print("⚠ chat_sft.py already patched")

    # Patch flash_attention.py: strip enable_gqa for PyTorch < 2.5 compat
    fa_path = os.path.join(repo_path, "nanochat", "flash_attention.py")
    with open(fa_path) as f:
        content = f.read()
    if "enable_gqa=enable_gqa" in content:
        content = content.replace(", enable_gqa=enable_gqa", "")
        with open(fa_path, "w") as f:
            f.write(content)
        print("✓ Patched flash_attention.py: stripped enable_gqa for PyTorch < 2.5")
    else:
        print("⚠ flash_attention.py already patched or pattern changed")

    print("nanochat setup complete")


# ── SFT (self-contained: handles checkpoint + tokenizer internally) ────────


@app.function(
    image=image,
    volumes={str(VOLUME_DIR): volume},
    gpu="A10G:2",
    memory=32768,
    timeout=14400,
)
def run_sft():
    """Run SFT for 1500 steps on 2x A10G.

    Self-contained: downloads checkpoint + trains tokenizer if needed.
    """
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    import urllib.request

    sys.path.insert(0, "/nanochat")
    setup_nanochat()

    os.environ["NANOCHAT_BASE_DIR"] = str(VOLUME_DIR)
    os.environ["WANDB_RUN"] = "dummy"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

    # ── Step 0: Download checkpoint ──
    if not (VOLUME_CHECKPOINT_DIR / "model_008600.pt").exists():
        print("\n=== Step 0: Downloading checkpoint ===")
        tmp = Path(tempfile.mkdtemp())
        clone_dir = tmp / "checkpoint_repo"
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/rajofearth/temp-file-share.git",
                str(clone_dir),
            ],
            check=True,
            capture_output=True,
        )
        print("Pulling LFS objects...")
        subprocess.run(
            ["git", "-C", str(clone_dir), "lfs", "pull"],
            check=True,
            capture_output=True,
        )
        VOLUME_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            str(clone_dir / "model_008600.pt"),
            str(VOLUME_CHECKPOINT_DIR / "model_008600.pt"),
        )
        shutil.copy2(
            str(clone_dir / "meta_008600.json"),
            str(VOLUME_CHECKPOINT_DIR / "meta_008600.json"),
        )
        shutil.rmtree(tmp, ignore_errors=True)
        volume.commit()
        print("✓ Checkpoint ready")

    # ── Step 1: Train tokenizer ──
    if not (TOKENIZER_DIR / "tokenizer.pkl").exists():
        print("\n=== Step 1: Training tokenizer ===")
        env = os.environ.copy()
        env["PYTHONPATH"] = "/nanochat"

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        existing = len(list(DATA_DIR.glob("*.parquet")))
        if existing < 10:
            print("Downloading 10 data shards...")
            subprocess.run(
                ["python", "-m", "nanochat.dataset", "-n", "10"],
                cwd="/nanochat",
                env=env,
                check=True,
                capture_output=True,
            )
            climbmix = VOLUME_DIR / "base_data_climbmix"
            if climbmix.exists():
                for f in climbmix.glob("*.parquet"):
                    shutil.move(str(f), str(DATA_DIR / f.name))

        print("Training tokenizer (~2-3 min)...")
        result = subprocess.run(
            ["python", "scripts/tok_train.py", "--max-chars=2000000000"],
            cwd="/nanochat",
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"tok_train.py stdout:\n{result.stdout[-2000:]}")
            print(f"tok_train.py stderr:\n{result.stderr[-2000:]}")
            result.check_returncode()
        volume.commit()
        print("✓ Tokenizer ready")

    # ── Step 2: Download identity conversations ──
    if not IDENTITY_CONVOS.exists():
        print("\n=== Downloading identity conversations ===")
        urllib.request.urlretrieve(
            "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl",
            str(IDENTITY_CONVOS),
        )
        volume.commit()

    # ── Step 3: Run SFT ──
    print("\n" + "=" * 60)
    print("Starting SFT on 2× A10G")
    print(f"  Steps: {NUM_ITERATIONS}")
    print(
        f"  Batch: {DEVICE_BATCH_SIZE}×{MAX_SEQ_LEN}×2 GPUs = {DEVICE_BATCH_SIZE * MAX_SEQ_LEN * 2} tok/microbatch"
    )
    print(f"  Grad accum: {TOTAL_BATCH_SIZE // (DEVICE_BATCH_SIZE * MAX_SEQ_LEN * 2)}")
    print(f"  Total tokens/step: {TOTAL_BATCH_SIZE:,}")
    print(f"  load_optimizer=0 | eval_every={EVAL_EVERY}")
    print("=" * 60)

    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "-m",
        "scripts.chat_sft",
        "--",  # ← CRITICAL: separates torchrun args from module args (avoids --run ambiguity)
        "--max-seq-len",
        str(MAX_SEQ_LEN),
        "--device-batch-size",
        str(DEVICE_BATCH_SIZE),
        "--total-batch-size",
        str(TOTAL_BATCH_SIZE),
        "--num-iterations",
        str(NUM_ITERATIONS * 32),  # 1500 steps × 32 grad_accum = 48000 micro-batches
        "--eval-every",
        str(EVAL_EVERY),
        "--eval-tokens",
        str(EVAL_TOKENS),
        "--init-lr-frac",
        "0.1",
        "--warmup-ratio",
        "0.1",
        "--warmdown-ratio",
        "0.5",
        "--chatcore-every",
        "0",  # disabled during training (too slow), run eval separately
        "--final-lr-frac",
        "0.0",
        "--load-optimizer",
        "0",
        "--run",
        "dummy",
    ]

    print(f"Running: {' '.join(cmd)}")
    sys.stdout.flush()

    # Run with stderr captured so we can see errors even if torchrun swallows them
    import subprocess as _sp

    _result = _sp.run(cmd, cwd="/nanochat", capture_output=True, text=True)

    # Print combined output
    if _result.stdout:
        print(_result.stdout[-5000:] if len(_result.stdout) > 5000 else _result.stdout)
    if _result.stderr:
        print("\n==== STDERR ====")
        print(_result.stderr[-5000:] if len(_result.stderr) > 5000 else _result.stderr)
        print("==== END STDERR ====\n")

    volume.commit()
    print(f"\nSFT exited with code {_result.returncode}")

    if _result.returncode != 0:
        raise RuntimeError(f"SFT failed with code {_result.returncode}")

    # Show output checkpoints
    print("\n✓ SFT checkpoints:")
    for f in sorted(SFT_CHECKPOINT_DIR.iterdir()):
        print(f"  {f.name} ({f.stat().st_size / (1024 * 1024):.1f} MB)")

    return _result.returncode


@app.function(
    image=image,
    volumes={str(VOLUME_DIR): volume},
    gpu="T4",
    timeout=300,
)
def test_inference(prompt: str = "What is 2 + 2?"):
    """Test the SFT model by generating a response."""
    import os
    import sys

    import torch

    sys.path.insert(0, "/nanochat")
    setup_nanochat()
    os.environ["NANOCHAT_BASE_DIR"] = str(VOLUME_DIR)

    from nanochat.checkpoint_manager import load_model_from_dir
    from nanochat.tokenizer import get_tokenizer

    device = torch.device("cuda")
    tokenizer = get_tokenizer()

    sft_dir = str(SFT_CHECKPOINT_DIR)
    parent_dir = str(SFT_CHECKPOINT_DIR.parent)  # /vol/chatsft_checkpoints
    tag = SFT_CHECKPOINT_DIR.name  # "d6"
    if not os.path.exists(sft_dir) or not any(
        f.endswith(".pt") for f in os.listdir(sft_dir)
    ):
        print(f"✗ No SFT checkpoint at {sft_dir}")
        # Try fallback path
        sft_dir = os.path.join("/nanochat", "chatsft_checkpoints", "d6")
        parent_dir = str(Path(sft_dir).parent)
        tag = "d6"
        if not os.path.exists(sft_dir):
            print(f"✗ No SFT checkpoint found anywhere")
            return

    print(f"Loading {tag} from {parent_dir}...")
    model, _, meta = load_model_from_dir(
        parent_dir, device=device, phase="eval", model_tag=tag
    )
    model.eval()
    print(f"✓ Loaded SFT step {meta.get('step', '?')}")

    ids, _ = tokenizer.render_conversation(
        {"messages": [{"role": "user", "content": prompt}]},
        max_tokens=MAX_SEQ_LEN,
    )
    x = torch.tensor([ids], dtype=torch.long, device=device)

    print(f"\nPrompt: {prompt}")
    print("Response: ", end="", flush=True)

    with torch.no_grad():
        for _ in range(300):
            logits = model(x)[:, -1, :]
            next_tok = torch.multinomial(torch.softmax(logits / 0.8, dim=-1), 1)
            x = torch.cat([x, next_tok], dim=1)
            print(tokenizer.decode([next_tok.item()]), end="", flush=True)
    print()
    return prompt


# ── Download SFT Checkpoint ────────────────────────────────────────────────


@app.function(
    image=image,
    volumes={str(VOLUME_DIR): volume},
    timeout=120,
)
def download_sft():
    """Download SFT checkpoint from volume to /nanochat_local_checkpoints."""
    import shutil

    volume.reload()

    if not SFT_CHECKPOINT_DIR.exists():
        print(f"✗ No checkpoint at {SFT_CHECKPOINT_DIR}")
        return

    local_dir = Path("/nanochat_local_checkpoints")
    local_dir.mkdir(parents=True, exist_ok=True)

    for f in SFT_CHECKPOINT_DIR.iterdir():
        dest = local_dir / f.name
        shutil.copy(str(f), str(dest))
        print(f"Downloaded: {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

    print(f"\n✓ Saved to {local_dir}")
    return [str(local_dir / f) for f in SFT_CHECKPOINT_DIR.iterdir()]


# ── Post-Training Evaluation ────────────────────────────────────────────


@app.function(
    image=image,
    volumes={str(VOLUME_DIR): volume},
    gpu="A10G:2",
    timeout=7200,
    memory=32768,
)
def run_eval():
    """Run ChatCORE evaluation on the SFT checkpoint (2 GPUs)."""
    import os
    import subprocess
    import sys

    sys.path.insert(0, "/nanochat")
    setup_nanochat()
    os.environ["NANOCHAT_BASE_DIR"] = str(VOLUME_DIR)
    os.environ["PYTHONUNBUFFERED"] = "1"

    if not SFT_CHECKPOINT_DIR.exists() or not any(
        SFT_CHECKPOINT_DIR.glob("model_*.pt")
    ):
        print(f"✗ No SFT checkpoint at {SFT_CHECKPOINT_DIR}")
        return

    print("\n" + "=" * 60)
    print("Running ChatCORE evaluation on SFT checkpoint")
    print("Tasks: ARC-Easy, ARC-Challenge, MMLU, GSM8K, HumanEval, SpellingBee")
    print("=" * 60)

    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "-m",
        "scripts.chat_eval",
        "--",
        "-i",
        "sft",
        "-a",
        "all",
        "-b",
        "8",
    ]
    result = subprocess.run(cmd, cwd="/nanochat")
    print(f"\nEval exited with code {result.returncode}")
    return result.returncode


# ── CLI Entrypoint ─────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(
    inference_prompt: str = "",
):
    """Run the full SFT pipeline on nanochat d6.

    Spawns both tokenizer training and SFT as a single detached job.
    """
    print("Spawning SFT on 2× A10G (detached, self-contained)...")
    run_sft.spawn()
    print()
    print("=" * 60)
    print("  SFT SPAWNED — running in the cloud.")
    print("  Monitor:    modal app logs nanochat-sft")
    print("  Download:   modal run nanochat_sft.py::download_sft")
    print("  Inference:  modal run nanochat_sft.py::test_inference --prompt 'Hi!'")
    print("  Eval:       modal run nanochat_sft.py::run_eval")
    print("  Estimated:  ~60-90 min, ~$1-2 total")
    print("=" * 60)
