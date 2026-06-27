#!/usr/bin/env zsh
#===============================================================================
# run_sft.sh — nanochat SFT on Modal helper
#
# Usage:
#   ./run_sft.sh              # run SFT synchronously (shows live logs)
#   ./run_sft.sh detach       # run SFT detached (background)
#   ./run_sft.sh logs         # tail logs from latest SFT run
#   ./run_sft.sh infer        # test inference on SFT checkpoint
#   ./run_sft.sh eval         # run ChatCORE evaluation
#   ./run_sft.sh download     # download SFT checkpoint
#   ./run_sft.sh status       # check volume contents
#===============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Activate root-level modal venv
VENV_DIR="$(dirname "$SCRIPT_DIR")/.venv"
export PATH="$HOME/.local/bin:$PATH"
source "$VENV_DIR/bin/activate" 2>/dev/null || {
    echo "✗ No .venv found. Run: uv venv && source .venv/bin/activate && uv pip install modal"
    exit 1
}

CMD="${1:-run}"

case "$CMD" in
    run)
        echo "━━━ Running SFT synchronously (live logs) ━━━"
        echo "    Ctrl+C to cancel. Checkpoints are saved on volume."
        echo ""
        modal run nanochat_sft.py::run_sft
        ;;
    detach|detached)
        echo "━━━ Spawning detached SFT ━━━"
        modal run --detach nanochat_sft.py
        ;;
    logs)
        echo "━━━ Tailing logs ━━━"
        APP_ID=$(modal app list --json 2>/dev/null | python3 -c "
import json, sys
apps = json.load(sys.stdin)
active = [a for a in apps if a.get('state') == 'running']
if active:
    print(active[0]['app_id'])
else:
    print('')
" 2>/dev/null)
        if [ -z "$APP_ID" ]; then
            # Try to get the latest app
            APP_ID=$(modal app list --json 2>/dev/null | python3 -c "
import json, sys
apps = json.load(sys.stdin)
if apps:
    print(apps[-1]['app_id'])
else:
    print('')
" 2>/dev/null)
        fi
        if [ -z "$APP_ID" ]; then
            echo "✗ No apps found. Start one with: ./run_sft.sh"
        else
            echo "Tailing logs for app $APP_ID ..."
            modal app logs "$APP_ID"
        fi
        ;;
    infer|inference)
        PROMPT="${2:-What is 2 + 2?}"
        echo "━━━ Testing inference ━━━"
        echo "Prompt: $PROMPT"
        modal run nanochat_sft.py::test_inference --prompt "$PROMPT"
        ;;
    eval|evaluate)
        echo "━━━ Running ChatCORE evaluation ━━━"
        echo "    This runs on 2× A10G and may take 30-60 min."
        modal run nanochat_sft.py::run_eval
        ;;
    download)
        echo "━━━ Downloading SFT checkpoint ━━━"
        modal run nanochat_sft.py::download_sft
        echo ""
        echo "To fetch to local machine:"
        echo "  modal volume get nanochat-vol /chatsft_checkpoints/d6/model_001500.pt ./my_sft_model.pt"
        ;;
    status)
        echo "━━━ Volume status ━━━"
        echo "--- Checkpoints ---"
        modal volume ls nanochat-vol /base_checkpoints/d6/ 2>/dev/null || echo "  (empty)"
        echo ""
        echo "--- Tokenizer ---"
        modal volume ls nanochat-vol /tokenizer/ 2>/dev/null || echo "  (empty)"
        echo ""
        echo "--- SFT output ---"
        modal volume ls nanochat-vol /chatsft_checkpoints/d6/ 2>/dev/null || echo "  (empty)"
        ;;
    *)
        echo "Usage: $0 [run|detach|logs|infer|eval|download|status]"
        echo ""
        echo "Commands:"
        echo "  run        Run SFT synchronously (live logs, recommended for first run)"
        echo "  detach     Run SFT in background (doesn't stop if you close terminal)"
        echo "  logs       Tail logs from latest/active SFT run"
        echo "  infer      Test inference with a prompt"
        echo "  eval       Run ChatCORE evaluation on SFT checkpoint"
        echo "  download   Download SFT checkpoint from volume"
        echo "  status     Check volume contents"
        ;;
esac
