"""FastAPI inference service for jatan-ai.

Startup:
    CUDA_VISIBLE_DEVICES=1 uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Environment variables (or .env):
    JATAN_CHECKPOINT        path to road classifier checkpoint (default: checkpoints/best_model.pt)
    JATAN_BRIDGE_CHECKPOINT path to bridge seg checkpoint (default: checkpoints/bridge_seg_best.pt)
    JATAN_ADAPTER           path to VLM LoRA adapter (optional — disables VLM if unset)
    JATAN_DEVICE            cuda device string e.g. "cuda:1" (default: auto)
"""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image


# ---------------------------------------------------------------------------
# Lifespan — load models once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.api.inference_service import InferenceService

    bridge_checkpoint = os.getenv("JATAN_BRIDGE_CHECKPOINT", "checkpoints/bridge_seg_best.pt")
    adapter_path      = os.getenv("JATAN_ADAPTER",           None)
    device            = os.getenv("JATAN_DEVICE",            None)

    logger.info("Loading InferenceService (bridge_checkpoint={}, adapter={})", bridge_checkpoint, adapter_path)
    app.state.service = InferenceService(
        bridge_checkpoint=bridge_checkpoint,
        adapter_path=adapter_path,
        device=device,
    )
    logger.info("InferenceService ready.")
    yield


app = FastAPI(
    title="Jatan AI — Infrastructure Damage Inference",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok", "vlm_loaded": app.state.service._bridge_vlm is not None}


@app.post("/infer")
async def infer(
    images: list[UploadFile] = File(..., description="One or more images (JPEG/PNG)"),
    use_vlm: bool = Form(True, description="Include VLM triage report (default: true)"),
    threshold: float = Form(0.5, description="Detection confidence threshold (default: 0.5)"),
):
    """Run damage inference on one or more uploaded images.

    Returns the same JSON structure as the CLI `infer` command:
    {
      "images": [{domain, bridge_seg, severity, passability, reasoning?}],
      "aggregate": {severity, passability}
    }
    """
    pil_images: list[Image.Image] = []
    for upload in images:
        raw = await upload.read()
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail=f"Cannot decode image: {upload.filename}")
        pil_images.append(img)

    if not pil_images:
        raise HTTPException(status_code=400, detail="No valid images provided.")

    service = app.state.service
    result = service.run(pil_images, use_vlm=use_vlm, threshold=threshold)
    return JSONResponse(content=result)
