#!/usr/bin/env bash
# Usage: bash deploy/start.sh [--adapter <path>] [--port <port>] [--device <cuda:N>]
# Starts uvicorn + cloudflared quick tunnel and prints the public URL.

set -e

PORT=8000
ADAPTER=""
DEVICE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --adapter) ADAPTER="$2"; shift 2 ;;
        --port)    PORT="$2";    shift 2 ;;
        --device)  DEVICE="$2";  shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── env vars ───────────────────────────────────────────────────────────────────
export JATAN_BRIDGE_CHECKPOINT="${JATAN_BRIDGE_CHECKPOINT:-checkpoints/bridge_seg_best.pt}"
[ -n "$ADAPTER" ] && export JATAN_ADAPTER="$ADAPTER"
[ -n "$DEVICE"  ] && export JATAN_DEVICE="$DEVICE"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error

# ── check cloudflared (system install or local binary) ────────────────────────
if command -v cloudflared &>/dev/null; then
    CLOUDFLARED=cloudflared
elif [ -x "./cloudflared" ]; then
    CLOUDFLARED=./cloudflared
else
    echo "[error] cloudflared not found. Download the binary (no sudo needed):"
    echo "  wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared"
    echo "  chmod +x cloudflared"
    exit 1
fi

# ── start uvicorn in background ────────────────────────────────────────────────
LOG_FILE="logs/server.log"
mkdir -p logs

echo "[1/2] Starting uvicorn on port $PORT... (logs → $LOG_FILE)"
uv run uvicorn src.api.app:app \
    --host 127.0.0.1 \
    --port "$PORT" \
    --log-level info \
    2>&1 | tee -a "$LOG_FILE" &
UVICORN_PID=$!

# wait for uvicorn to be ready
for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:$PORT/health" &>/dev/null; then
        echo "      uvicorn ready (pid $UVICORN_PID)"
        break
    fi
    sleep 1
done

# ── start cloudflared quick tunnel ────────────────────────────────────────────
echo "[2/2] Starting cloudflared quick tunnel..."
echo "      Public URL will appear below (look for trycloudflare.com):"
echo "──────────────────────────────────────────────────────────────────"
$CLOUDFLARED tunnel --url "http://127.0.0.1:$PORT" &
CLOUDFLARED_PID=$!

# ── shutdown on Ctrl-C ─────────────────────────────────────────────────────────
trap "echo; echo 'Shutting down...'; kill $UVICORN_PID $CLOUDFLARED_PID 2>/dev/null; exit 0" INT TERM

wait
