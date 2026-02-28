# Jatan AI

Post-disaster bridge and road damage assessment using segmentation + VLM reasoning.

**Segmentation:** SegFormer-B5 (3-class: Undamaged / Damaged / Destroyed) + DPT-Large depth weighting

**VLM:** Qwen3-VL-2B-Instruct with LoRA — trained via SFT and GRPO (RISE-R1)

## Installation

```bash
uv sync
```

## Environment

Create `.env`:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
OPENROUTER_API_KEY=your_key      # required for generate-annotations
```

## CLI Commands

```bash
# Train bridge segmenter (EIDSeg, 3-class)
uv run main.py train --task eidseg-bridge [--epochs1 10] [--epochs2 20]

# Evaluate bridge segmenter on EIDSeg validation set
uv run main.py eval [--checkpoint checkpoints/bridge_seg_best.pt]

# Inference (outputs JSON with severity + passability)
uv run main.py infer image.jpg [--checkpoint ...] [--with-reasoning --vlm-adapter ...]

# Generate VLM training annotations via OpenRouter API
uv run main.py generate-annotations [--cot] [--model ...]

# LoRA finetune Qwen3-VL-2B-Instruct on annotations
uv run main.py train-vlm --annotations data/vlm_annotations.jsonl [--cot]

# GRPO reinforcement refinement (requires torchrun)
torchrun --nproc_per_node=1 main.py train-vlm-grpo --annotations data/vlm_cot_annotations.jsonl
```

## API Server

```bash
JATAN_BRIDGE_CHECKPOINT=checkpoints/bridge_seg_best.pt \
JATAN_ADAPTER=checkpoints/vlm_lora/grpo/final_adapter \
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Endpoints: `GET /health`, `POST /infer` (multipart images), `POST /overlay`

## Checkpoints

| File | Description |
|------|-------------|
| `checkpoints/bridge_seg_best.pt` | EIDSeg bridge segmenter |
| `checkpoints/vlm_lora/final_adapter/` | LoRA SFT adapter |
| `checkpoints/vlm_lora/grpo/final_adapter/` | GRPO-refined adapter |

## Output Format

```json
{
  "images": [
    {
      "image": "path",
      "seg": { "presence": ["Damaged"], "coverage": {...} },
      "severity": { "score": 0.42, "label": "moderate" },
      "passability": "bike_only",
      "reasoning": { "report": "...", "detected": [...] }
    }
  ],
  "aggregate": { "severity": {...}, "passability": "bike_only" }
}
```

Passability tiers: `possible` (Bisa) · `bike_only` (Roda-2) · `impossible` (Tidak Bisa)
Aggregate uses worst-case across all images (safety-first).

## Models
The trained models can be accessed here:
https://drive.google.com/drive/folders/1qSOkawg_yZti5R3nVl_M7Vyhwcj56Rd3?usp=drive_link
