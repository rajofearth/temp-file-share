"""Download a Hugging Face model snapshot to /vol/bases/."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from volumes import DEFAULT_MODEL_ID, base_model_dir, ensure_volume_layout


def download_base(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    token: str | None = None,
    force: bool = False,
) -> Path:
    dest = base_model_dir(model_id)
    ensure_volume_layout()
    dest.mkdir(parents=True, exist_ok=True)

    marker = dest / ".download_complete"
    if marker.exists() and not force and any(dest.iterdir()):
        print(f"Model already present at {dest}")
        return dest

    print(f"Downloading {model_id} → {dest}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        token=token,
    )
    marker.touch()
    print(f"Download complete: {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LFM weights to /vol/bases/")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_base(args.model_id, force=args.force)


if __name__ == "__main__":
    main()
