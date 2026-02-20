import argparse
import sys
import os

from bikit.utils import download_dataset
from loguru import logger

def cmd_download(args: argparse.Namespace) -> None:
    """Download dacl1k (bridge) and RDD2022 (road) datasets."""
    data_root: str = args.data_root

    logger.info("Downloading dacl1k (bridge dataset)")
    download_dataset("dacl1k", cache_dir=data_root)
    logger.success("dacl1k ready.")

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
    from src.trainer import Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on {}", device)
    model = JatanMTL().to(device)
    trainer = Trainer(
        model,
        device,
        data_root=args.data_root,
        batch_size=args.batch_size,
        epochs1=args.epochs1,
        epochs2=args.epochs2,
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

    from src.dataset import _BRIDGE_CLASSES, _ROAD_CLASSES

    logger.info(
        "Evaluation Results\n"
        "  val_loss:         {:.4f}\n"
        "  bridge_macro_f1:  {:.4f}\n"
        "  road_macro_f1:    {:.4f}",
        metrics["val_loss"],
        metrics["bridge_macro_f1"],
        metrics["road_macro_f1"],
    )


def cmd_infer(args: argparse.Namespace) -> None:
    logger.warning("Inference not yet implemented.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jatan-ai",
        description="Jatan AI — Multi-task learning for road and bridge damage detection",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # download
    p_dl = sub.add_parser("download", help="Download dacl1k and RDD2022 datasets")
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
    p_tr = sub.add_parser("train", help="Train the MTL model")
    p_tr.add_argument("--batch-size", type=int, default=32)
    p_tr.add_argument("--epochs1",    type=int, default=10,
                      help="Epochs for frozen-backbone phase")
    p_tr.add_argument("--epochs2",    type=int, default=20,
                      help="Epochs for full fine-tune phase")
    p_tr.add_argument("--data-root",  default="data/raw")
    p_tr.set_defaults(func=cmd_train)

    # eval
    p_ev = sub.add_parser("eval", help="Evaluate on validation set")
    p_ev.add_argument("--data-root",   default="data/raw")
    p_ev.add_argument("--batch-size",  type=int, default=32)
    p_ev.add_argument("--checkpoint",  default="checkpoints/best_model.pt",
                      help="Path to checkpoint file (default: checkpoints/best_model.pt)")
    p_ev.set_defaults(func=cmd_eval)

    # infer
    p_in = sub.add_parser("infer", help="Run inference on a single image")
    p_in.add_argument("image_path")
    p_in.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p_in.add_argument("--threshold",  type=float, default=0.5)
    p_in.set_defaults(func=cmd_infer)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
