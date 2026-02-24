import argparse
import sys
import os

from loguru import logger


def cmd_download(args: argparse.Namespace) -> None:
    """Download RDD2022 (road) dataset. dacl10k auto-downloads on first BridgeDataset init."""
    data_root: str = args.data_root

    logger.info(
        "dacl10k (bridge dataset) will be downloaded automatically on first "
        "`uv run main.py train --task bridge-seg` run via dacl10k-toolkit."
    )

    logger.info("Downloading RDD2022 (road dataset)")
    rdd_root = os.path.join(data_root, "rdd2022")

    if os.path.isdir(rdd_root) and os.listdir(rdd_root):
        logger.info("RDD2022 already exists at {}. Skipping.", rdd_root)
        return

    try:
        _download_rdd2022(rdd_root, args.kaggle_dataset)
    except Exception as exc:
        logger.error("RDD2022 download failed: {}", exc)
        sys.exit(1)


def _download_rdd2022(dest_dir: str, kaggle_dataset: str) -> None:
    """Download RDD2022 via the Kaggle Python API."""
    os.makedirs(dest_dir, exist_ok=True)
    logger.info("Downloading '{}' -> {}", kaggle_dataset, dest_dir)

    from dotenv import load_dotenv
    load_dotenv()

    import kaggle as kg
    kg.api.authenticate()
    kg.api.dataset_download_files(kaggle_dataset, path=dest_dir, unzip=True, quiet=False)
    logger.success("RDD2022 downloaded to {}.", dest_dir)


def cmd_train(args: argparse.Namespace) -> None:
    import torch
    from src.model import JatanMTL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on {}", device)
    model = JatanMTL().to(device)

    if args.task == "road":
        from src.trainer import Trainer
        trainer = Trainer(
            model,
            device,
            data_root=args.data_root,
            batch_size=args.batch_size,
            epochs1=args.epochs1,
            epochs2=args.epochs2,
        )
        trainer.run()

    elif args.task == "bridge-seg":
        from src.trainer import BridgeSegTrainer
        trainer = BridgeSegTrainer(
            model,
            device,
            data_root=args.data_root,
            batch_size=args.batch_size,
            epochs1=args.epochs1,
            epochs2=args.epochs2,
            resume_phase2=args.resume_phase2,
        )
        trainer.run()

    elif args.task == "eidseg-bridge":
        from src.trainer import EIDSegBridgeTrainer
        trainer = EIDSegBridgeTrainer(
            model,
            device,
            data_root=args.data_root,
            batch_size=args.batch_size,
            epochs1=args.epochs1,
            epochs2=args.epochs2,
            resume_phase2=args.resume_phase2,
        )
        trainer.run()


def cmd_eval(args: argparse.Namespace) -> None:
    import torch
    from src.dataset import CombinedDamageDataset, get_transform
    from src.model import JatanMTL
    from src.trainer import Trainer
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JatanMTL().to(device)

    checkpoint_path = getattr(args, "checkpoint", "checkpoints/best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded checkpoint from {} (epoch {})", checkpoint_path, checkpoint["epoch"])

    val_ds = CombinedDamageDataset(
        split="val", data_root=args.data_root, transform=get_transform("val")
    )
    num_workers = min(4, os.cpu_count() or 1)

    from src.trainer import custom_collate_fn
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    trainer = Trainer(model, device, data_root=args.data_root)
    metrics = trainer.validate(val_loader)

    logger.info(
        "Evaluation Results\n"
        "  val_loss:         {:.4f}\n"
        "  road_macro_f1:    {:.4f}",
        metrics["val_loss"],
        metrics["road_macro_f1"],
    )


def _severity_label(score: float) -> str:
    if score < 0.2:
        return "Ringan"
    elif score < 0.5:
        return "Sedang"
    return "Berat"


def _aggregate_passability(labels: list[str]) -> str:
    """Safety-first (worst-case) aggregation across multiple images."""
    if "Tidak Bisa" in labels:
        return "Tidak Bisa"
    if "Roda-2" in labels:
        return "Roda-2"
    return "Bisa"


def cmd_generate_annotations(args: argparse.Namespace) -> None:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    if not args.api_key:
        args.api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not args.api_key:
        logger.error("No API key found. Set OPENROUTER_API_KEY in .env or pass --api-key.")
        sys.exit(1)

    if args.domain == "bridge":
        from src.vlm.annotation_generator import EIDSegAnnotationGenerator
        generator = EIDSegAnnotationGenerator(
            data_root=args.data_root,
            output_path=args.output,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            split=args.split,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            request_delay=args.request_delay,
            cot=args.cot,
            max_samples=args.max_samples,
        )
    else:
        from src.vlm.annotation_generator import RoadAnnotationGenerator
        generator = RoadAnnotationGenerator(
            data_root=args.data_root,
            output_path=args.output,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            split=args.split,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            request_delay=args.request_delay,
            cot=args.cot,
            max_samples=args.max_samples,
        )
    generator.run()


def cmd_train_vlm(args: argparse.Namespace) -> None:
    from src.vlm.trainer import VLMTrainer

    trainer = VLMTrainer(
        annotations_path=args.annotations,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        lr=args.lr,
        cot=args.cot,
    )
    trainer.run()


def cmd_train_vlm_grpo(args: argparse.Namespace) -> None:
    from src.vlm.grpo_trainer import GRPOVLMTrainer

    trainer = GRPOVLMTrainer(
        annotations_path=args.annotations,
        model_name=args.model_name,
        base_adapter=args.base_adapter,
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        per_device_batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
    )
    trainer.run()


def _pil_to_base64(img) -> str:
    import base64, io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def cmd_infer(args: argparse.Namespace) -> None:
    import json
    import torch
    from PIL import Image
    from torchvision import transforms

    from src.dataset import _BRIDGE_CLASSES
    from src.model import JatanMTL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JatanMTL().to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt, strict=False)
    logger.info("Loaded checkpoint from {}", args.checkpoint)

    model.eval()

    seg_tfm = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_paths = args.image_paths

    # Lazy-load VLM if reasoning is requested
    vlm = None
    if getattr(args, "with_reasoning", False):
        if not getattr(args, "vlm_adapter", None):
            logger.error("--with-reasoning requires --vlm-adapter.")
            sys.exit(1)
        from src.vlm.inference import BridgeVLMInference
        logger.info("Loading VLM adapter from {}", args.vlm_adapter)
        vlm = BridgeVLMInference(
            adapter_path=args.vlm_adapter,
            max_new_tokens=getattr(args, "max_new_tokens", 256),
        )

    per_image = []
    all_severities: list[float] = []
    all_passability: list[str] = []

    for image_path in image_paths:
        img = Image.open(image_path).convert("RGB")

        x_seg = seg_tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model.segment_bridge(x_seg, with_depth=True)

        class_map    = out["class_map"][0].cpu()
        depth_map    = out["depth_map"][0].cpu()
        total_pixels = class_map.numel()

        coverage = {
            cls: round(float((class_map == i).sum()) / total_pixels * 100, 2)
            for i, cls in enumerate(_BRIDGE_CLASSES)
        }
        detected_classes = [
            cls for cls in ["Damaged", "Destroyed"] if coverage.get(cls, 0.0) > 0
        ]

        severity_score = float(model.compute_bridge_severity(
            class_map.unsqueeze(0), depth_map.unsqueeze(0)
        )[0])
        passability    = model.compute_bridge_passability(severity_score)

        all_severities.append(severity_score)
        all_passability.append(passability)

        entry: dict = {
            "image":   image_path,
            "domain":  args.domain,
            "seg": {
                "presence": detected_classes,
                "coverage": coverage,
            },
            "severity":    {"score": round(severity_score, 4), "label": _severity_label(severity_score)},
            "passability": passability,
        }

        if vlm is not None:
            vlm_result = vlm.describe(
                img, class_map, severity_score, passability, coverage,
            )
            entry["reasoning"] = {
                "report":        vlm_result["report"],
                "detected":      vlm_result["detected"],
                "overlay_image": _pil_to_base64(vlm_result["overlay_image"]),
            }

        per_image.append(entry)

    if all_severities:
        agg_severity_score = max(all_severities)
        agg_passability = _aggregate_passability(all_passability)
        result = {
            "images":    per_image,
            "aggregate": {
                "severity":    {"score": round(agg_severity_score, 4), "label": _severity_label(agg_severity_score)},
                "passability": agg_passability,
            },
        }
    else:
        result = {"images": per_image}

    print(json.dumps(result, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jatan-ai",
        description="Jatan AI — Multi-task learning for road and bridge damage detection",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # download
    p_dl = sub.add_parser("download", help="Download RDD2022 dataset (dacl10k auto-downloads)")
    p_dl.add_argument(
        "--data-root",
        default="data/raw",
        help="Root directory for raw data (default: data/raw)",
    )
    p_dl.add_argument(
        "--kaggle-dataset",
        default="aliabdelmenam/rdd-2022",
        help="Kaggle dataset slug for RDD2022 (default: aliabdelmenam/rdd-2022)",
    )
    p_dl.set_defaults(func=cmd_download)

    # train
    p_tr = sub.add_parser("train", help="Train the model")
    p_tr.add_argument(
        "--task",
        choices=["road", "bridge-seg", "eidseg-bridge"],
        default="road",
        help="Training task: 'road' (ResNet50), 'bridge-seg' (dacl10k 19-class), or 'eidseg-bridge' (EIDSeg 3-class)",
    )
    p_tr.add_argument("--batch-size", type=int, default=32)
    p_tr.add_argument("--epochs1",    type=int, default=10,
                      help="Epochs for frozen-backbone/encoder phase")
    p_tr.add_argument("--epochs2",    type=int, default=20,
                      help="Epochs for full fine-tune phase")
    p_tr.add_argument("--data-root",  default="data/raw",
                      help="Data root (use data/dacl10k for bridge-seg, data/raw/eidseg for eidseg-bridge)")
    p_tr.add_argument("--resume-phase2", action="store_true",
                      help="Skip Phase 1 and load checkpoints/bridge_seg_best.pt directly into Phase 2")
    p_tr.set_defaults(func=cmd_train)

    # eval
    p_ev = sub.add_parser("eval", help="Evaluate road model on validation set")
    p_ev.add_argument("--data-root",   default="data/raw")
    p_ev.add_argument("--batch-size",  type=int, default=32)
    p_ev.add_argument("--checkpoint",  default="checkpoints/best_model.pt",
                      help="Path to checkpoint file (default: checkpoints/best_model.pt)")
    p_ev.set_defaults(func=cmd_eval)

    # infer
    p_in = sub.add_parser("infer", help="Run inference on one or more images")
    p_in.add_argument("image_paths", nargs="+", help="Path(s) to input image(s)")
    p_in.add_argument(
        "--domain",
        choices=["bridge", "road"],
        default="bridge",
        help="Domain label for output JSON (both use EIDSeg 3-class segmentation)",
    )
    p_in.add_argument("--checkpoint", default="checkpoints/best_model.pt",
                      help="JatanMTL checkpoint (default: checkpoints/best_model.pt)")
    p_in.add_argument("--with-reasoning", action="store_true",
                      help="Run VLM reasoning after segmentation (requires --vlm-adapter)")
    p_in.add_argument("--vlm-adapter", default=None,
                      help="Path to LoRA adapter dir (e.g. checkpoints/vlm_lora/grpo/final_adapter)")
    p_in.add_argument("--max-new-tokens", type=int, default=256,
                      help="Max new tokens for VLM generation (default: 256)")
    p_in.set_defaults(func=cmd_infer)

    # generate-annotations
    p_ga = sub.add_parser("generate-annotations",
                          help="Generate VLM training annotations via vision API")
    p_ga.add_argument("--domain", choices=["bridge", "road"], default="bridge",
                      help="Domain to annotate: bridge (dacl10k) or road (RDD2022) (default: bridge)")
    p_ga.add_argument("--data-root",     default="data/raw/eidseg",
                      help="Data root — data/raw/eidseg for bridge, data/raw for road (default: data/raw/eidseg)")
    p_ga.add_argument("--output",        default="data/vlm_annotations.jsonl",
                      help="Output JSONL path (default: data/vlm_annotations.jsonl)")
    p_ga.add_argument("--api-key",       default=None,
                      help="API key (default: OPENROUTER_API_KEY from .env)")
    p_ga.add_argument("--base-url",      default="https://openrouter.ai/api/v1",
                      help="Base URL for the API (default: OpenRouter)")
    p_ga.add_argument("--model",         default="qwen/qwen2.5-vl-72b-instruct",
                      help="Vision model to use (default: qwen/qwen2.5-vl-72b-instruct)")
    p_ga.add_argument("--split",         default="train",
                      help="Dataset split to annotate (default: train)")
    p_ga.add_argument("--max-retries",   type=int,   default=3)
    p_ga.add_argument("--retry-delay",   type=float, default=5.0,
                      help="Base retry delay in seconds (doubles on each attempt)")
    p_ga.add_argument("--request-delay", type=float, default=1.0,
                      help="Sleep between successful requests in seconds (default: 1.0)")
    p_ga.add_argument("--cot", action="store_true",
                      help="Generate chain-of-thought reasoning traces (<think>...</think> + annotation)")
    p_ga.add_argument("--max-samples", type=int, default=None,
                      help="Stop after writing this many annotations (useful for limiting cost)")
    p_ga.set_defaults(func=cmd_generate_annotations)

    # train-vlm
    p_tv = sub.add_parser("train-vlm",
                          help="LoRA-finetune Qwen3-VL-2B-Instruct on bridge annotations")
    p_tv.add_argument("--annotations",  default="data/vlm_annotations.jsonl",
                      help="Path to JSONL annotations (default: data/vlm_annotations.jsonl)")
    p_tv.add_argument("--output-dir",   default="checkpoints/vlm_lora",
                      help="Checkpoint output dir (default: checkpoints/vlm_lora)")
    p_tv.add_argument("--batch-size",   type=int,   default=4)
    p_tv.add_argument("--grad-accum",   type=int,   default=4,
                      help="Gradient accumulation steps (default: 4)")
    p_tv.add_argument("--epochs",       type=int,   default=3)
    p_tv.add_argument("--lr",           type=float, default=5e-5)
    p_tv.add_argument("--cot",          action="store_true",
                      help="Train with CoT reasoning traces (requires --cot annotations)")
    p_tv.set_defaults(func=cmd_train_vlm)

    # train-vlm-grpo
    p_grpo = sub.add_parser("train-vlm-grpo",
                             help="RISE-R1 GRPO RL refinement on CoT-annotated data (run via torchrun)")
    p_grpo.add_argument("--annotations",       default="data/vlm_cot_annotations.jsonl",
                        help="CoT JSONL from generate-annotations --cot (default: data/vlm_cot_annotations.jsonl)")
    p_grpo.add_argument("--model-name",        default="Qwen/Qwen3-VL-2B-Instruct")
    p_grpo.add_argument("--base-adapter",      default=None,
                        help="Path to CoT SFT LoRA adapter from train-vlm --cot (Stage 2 output)")
    p_grpo.add_argument("--output-dir",        default="checkpoints/vlm_lora/grpo")
    p_grpo.add_argument("--num-generations",   type=int, default=8,
                        help="GRPO rollouts per prompt (default: 8, matches RISE config)")
    p_grpo.add_argument("--max-prompt-length", type=int, default=1024)
    p_grpo.add_argument("--max-new-tokens",    type=int, default=512)
    p_grpo.add_argument("--batch-size",        type=int, default=1)
    p_grpo.add_argument("--grad-accum",        type=int, default=2)
    p_grpo.add_argument("--lr",                type=float, default=2e-6)
    p_grpo.add_argument("--max-steps",         type=int, default=200)
    p_grpo.add_argument("--save-steps",        type=int, default=10)
    p_grpo.set_defaults(func=cmd_train_vlm_grpo)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
