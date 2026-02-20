# Jatan AI

Multi-task ResNet50 for infrastructure damage assessment on bridge (dacl1k) and road (RDD2022) datasets.

## Installation

```bash
uv sync
```

## Kaggle Credentials

Create `.env` file:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

Get API key from https://www.kaggle.com/settings

## Usage

```bash
# Download datasets (requires Kaggle credentials in .env)
uv run main.py download [--data-root data/raw]

# Train model
uv run main.py train [--batch-size 32] [--epochs1 10] [--epochs2 20]

# Evaluate
uv run main.py eval [--checkpoint checkpoints/best_model.pt]
```

## Commands

| Command | Description |
|---------|-------------|
| `download` | Download dacl1k and RDD2022 datasets |
| `train` | Train the multi-task model |
| `eval` | Evaluate on validation set |
| `infer` | Run inference (not yet implemented) |
