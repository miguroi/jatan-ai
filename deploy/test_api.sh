#!/usr/bin/env bash
# Quick smoke test against a running instance.
# Usage: bash deploy/test_api.sh <base-url> <image-path>
# Example (local):  bash deploy/test_api.sh http://localhost:8000 sample/jembatan_rusak_2.jpg
# Example (tunnel): bash deploy/test_api.sh https://xxxx.trycloudflare.com sample/jembatan_rusak_2.jpg

BASE_URL="${1:-http://localhost:8000}"
IMAGE="${2:-sample/jembatan_rusak_2.jpg}"

if [ ! -f "$IMAGE" ]; then
    echo "[error] Image not found: $IMAGE"
    exit 1
fi

echo "=== /health ==="
curl -sf "$BASE_URL/health" | python3 -m json.tool

echo ""
echo "=== /infer (bridge, no VLM) ==="
curl -sf "$BASE_URL/infer" \
    -F "images=@$IMAGE" \
    -F "domain=bridge" \
    -F "use_vlm=false" \
    | python3 -m json.tool
