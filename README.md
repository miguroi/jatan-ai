# Jatan AI

Bridge and road damage assessment is a critical bottleneck in post-disaster emergency response. Jatan AI automates severity triage and passability classification from field photos, enabling faster mobilization decisions for rescue teams.

**3rd Place — Hackvidia ITB 2026**

## Architecture

```
Field Photo
  └─ SegFormer-B2 encoder → SegFormer-B5 decoder   # damage segmentation (3 classes)
       + DPT-Large depth map                         # depth-weighted severity scoring
  └─ Qwen3-VL-2B-Instruct + LoRA                   # natural language damage report
       trained via SFT → GRPO (RISE-R1)             # format + passability + grounding rewards
```

Two-phase training: frozen encoder → full fine-tune with differential LR.

## Results

### Segmentation Model (SegFormer-B2 × EIDSeg)

| Metric | Score |
|--------|-------|
| mIoU | 0.741 |
| IoU — Damaged | 0.745 |
| IoU — Destroyed | 0.794 |
| IoU — Undamaged | 0.685 |
| Val Loss | 0.322 |

### VLM (Qwen3-VL-2B + LoRA, GRPO-refined)

Trained via SFT then GRPO (RISE-R1) with three reward signals: output format, passability accuracy, and defect grounding (hallucination penalty). Formal benchmark pending.

## Sample Output

![Bridge damage overlay](overlay_jembatan_rusak_2.png)

```
severity_score  = 0.717   (Berat)
passability     = impossible
priority_score  = 0.788   (population: 1,200 — impact × urgency × feasibility)
```

```json
{
  "images": [{
    "image": "sample/jembatan_rusak_2.jpg",
    "seg": { "presence": ["Damaged", "Destroyed"], "coverage": { "Damaged": 0.41, "Destroyed": 0.31 } },
    "severity": { "score": 0.717, "label": "Berat" },
    "passability": "impossible",
    "reasoning": { "report": "Bridge shows severe structural damage with visible cracking and exposed rebar. Crossing is not safe for any vehicle type.", "detected": ["spalling", "exposed_rebar", "crack"] }
  }],
  "aggregate": { "severity": { "score": 0.717, "label": "Berat" }, "passability": "impossible" }
}
```

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
# Train
uv run main.py train --task eidseg-bridge --epochs1 10 --epochs2 20 --checkpoint-name bridge_seg_best.pt

# Eval
uv run main.py eval --checkpoint checkpoints/bridge_seg_best.pt

# Infer (no VLM)
uv run main.py infer sample/jembatan_rusak_2.jpg --checkpoint checkpoints/bridge_seg_best.pt

# Infer (with VLM)
uv run main.py infer sample/jembatan_rusak_2.jpg --checkpoint checkpoints/bridge_seg_best.pt --with-reasoning --vlm-adapter checkpoints/vlm_lora/grpo_eidseg_2/final_adapter

# Generate annotations
uv run main.py generate-annotations --cot --split train

# SFT
uv run main.py train-vlm --annotations data/vlm_cot_annotations.jsonl --cot

# GRPO
torchrun --nproc_per_node=1 main.py train-vlm-grpo --annotations data/vlm_cot_annotations.jsonl --base-adapter checkpoints/vlm_lora/eidseg_cot/final_adapter
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

Download: [Google Drive](https://drive.google.com/drive/folders/1qSOkawg_yZti5R3nVl_M7Vyhwcj56Rd3?usp=drive_link)

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

## Passability Tiers

| Tier | Label | Meaning |
|------|-------|---------|
| `possible` | Bisa | All vehicles |
| `bike_only` | Roda-2 | Motorcycles only |
| `impossible` | Tidak Bisa | Impassable |

Aggregate uses worst-case across all images (safety-first).
