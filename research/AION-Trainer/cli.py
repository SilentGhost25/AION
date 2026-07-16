"""
AION Trainer CLI

Commands:
    ingest              Parse documents into Knowledge Objects and build dataset
    train               Train or fine-tune the model
    benchmark           Evaluate model on benchmarks
    export              Export model for deployment
    continual           Incremental training on new data
    curriculum          Show curriculum schedule
    status              Show training status
"""

import argparse
import sys
import os
import yaml
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("aion.cli")


def cmd_ingest(args):
    """Parse documents into Knowledge Objects and build training dataset."""
    from preprocessing.parallel_parser import ParallelParser
    from dataset.builder import DatasetBuilder
    from dataset.version import DatasetVersion

    logger.info("=" * 60)
    logger.info("AION Trainer — Dataset Build Mode")
    logger.info("=" * 60)

    input_paths = []
    if args.books:
        for p in args.books.split(","):
            path = Path(p.strip())
            if path.is_file():
                input_paths.append(path)
            else:
                input_paths.extend(path.glob("**/*"))
    if args.papers:
        for p in args.papers.split(","):
            path = Path(p.strip())
            if path.is_file():
                input_paths.append(path)
            else:
                input_paths.extend(path.glob("**/*"))
    if args.notes:
        for p in args.notes.split(","):
            path = Path(p.strip())
            if path.is_file():
                input_paths.append(path)
            else:
                input_paths.extend(path.glob("**/*"))

    input_paths = [str(p) for p in input_paths if p.suffix.lower() in (".pdf", ".docx", ".pptx", ".png", ".jpg")]
    logger.info(f"Found {len(input_paths)} documents to process")

    if not input_paths:
        logger.error("No documents found. Check your paths.")
        sys.exit(1)

    # Parse documents
    parser = ParallelParser(
        num_workers=args.workers,
        batch_size=args.batch_size,
        subject_code=args.subject,
    )
    knowledge_objects = parser.parse_all(input_paths)
    logger.info(f"Extracted {len(knowledge_objects)} Knowledge Objects")

    # Build dataset
    builder = DatasetBuilder(
        output_dir=args.output,
        subject_code=args.subject,
    )
    dataset_path = builder.build(knowledge_objects)

    # Version the dataset
    version = DatasetVersion(args.output)
    version_info = version.create_version(
        subject_code=args.subject,
        source_files=[str(p) for p in input_paths],
        num_objects=len(knowledge_objects),
    )

    logger.info(f"Dataset built: {dataset_path}")
    logger.info(f"Version: {version_info['version']}")
    logger.info("Ingestion complete.")


def cmd_train(args):
    """Train or fine-tune the model."""
    from trainer.aion_trainer import AIONTrainer

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Override config with CLI args
    if args.dataset:
        config["dataset"]["path"] = args.dataset
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.resume:
        config["training"]["resume_from"] = args.resume

    logger.info("=" * 60)
    logger.info("AION Trainer — Training Mode")
    logger.info("=" * 60)
    logger.info(f"Config: {config_path}")

    trainer = AIONTrainer(config)
    trainer.train()


def cmd_benchmark(args):
    """Evaluate model on benchmarks."""
    from benchmarks.evaluator import BenchmarkEvaluator
    from checkpoints.manager import CheckpointManager

    # Load config
    config = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)

    # Find model checkpoint
    ckpt_dir = config.get("checkpoints", {}).get("dir", "checkpoints")
    ckpt_manager = CheckpointManager(ckpt_dir)
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = ckpt_manager.get_latest()
        if not checkpoint_path:
            logger.error("No checkpoint found. Train first or specify --checkpoint.")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("AION Trainer — Benchmark Mode")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {checkpoint_path}")

    evaluator = BenchmarkEvaluator(config.get("benchmark", {}))
    results = evaluator.evaluate(checkpoint_path, subject_code=args.subject)

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    for metric, score in results.items():
        status = "[PASS]" if score >= 0.8 else "[FAIL]"
        print(f"  {status:<8} {metric:.<40} {score:.4f}")
    print("=" * 60)


def cmd_export(args):
    """Export model for deployment."""
    from checkpoints.manager import CheckpointManager
    from models.model_registry import ModelRegistry

    config = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)

    ckpt_dir = config.get("checkpoints", {}).get("dir", "checkpoints")
    ckpt_manager = CheckpointManager(ckpt_dir)

    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = ckpt_manager.get_latest()
        if not checkpoint_path:
            logger.error("No checkpoint found.")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("AION Trainer — Export Mode")
    logger.info("=" * 60)

    registry = ModelRegistry()
    export_path = registry.export(
        checkpoint_path=checkpoint_path,
        output_dir=args.output,
        format=args.format,
    )

    logger.info(f"Model exported to: {export_path}")


def cmd_continual(args):
    """Incremental training on new data."""
    from trainer.continual import ContinualLearner

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.dataset:
        config["dataset"]["path"] = args.dataset

    logger.info("=" * 60)
    logger.info("AION Trainer — Continual Learning Mode")
    logger.info("=" * 60)

    learner = ContinualLearner(config)
    learner.incremental_train()


def cmd_curriculum(args):
    """Show curriculum schedule."""
    from curriculum.scheduler import CurriculumScheduler

    config = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)

    scheduler = CurriculumScheduler(config.get("curriculum", {}))
    schedule = scheduler.get_schedule()

    print("\n" + "=" * 60)
    print("CURRICULUM SCHEDULE")
    print("=" * 60)
    for stage in schedule:
        print(f"\n  Stage {stage['stage']}: {stage['name']}")
        print(f"    Difficulty: {stage['difficulty']}")
        print(f"    Bloom Levels: {', '.join(stage['bloom_levels'])}")
        print(f"    Sample Types: {', '.join(stage['sample_types'])}")
        print(f"    Estimated Samples: {stage['num_samples']}")
    print("=" * 60)


def cmd_status(args):
    """Show training status."""
    from checkpoints.manager import CheckpointManager
    from dataset.version import DatasetVersion

    config = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)

    print("\n" + "=" * 60)
    print("AION TRAINER STATUS")
    print("=" * 60)

    # Checkpoints
    ckpt_dir = config.get("checkpoints", {}).get("dir", "checkpoints")
    ckpt_manager = CheckpointManager(ckpt_dir)
    latest = ckpt_manager.get_latest()
    if latest:
        info = ckpt_manager.get_info(latest)
        print(f"\n  Latest Checkpoint: {latest}")
        print(f"    Version: {info.get('version', 'unknown')}")
        print(f"    Epoch: {info.get('epoch', 'unknown')}")
        print(f"    Score: {info.get('score', 'unknown')}")
    else:
        print("\n  No checkpoints found.")

    # Dataset
    dataset_dir = config.get("dataset", {}).get("path", "dataset")
    if Path(dataset_dir).exists():
        version = DatasetVersion(dataset_dir)
        versions = version.list_versions("BAI401")  # default subject check
        if versions:
            latest_version = versions[-1]
            print(f"\n  Latest Dataset: {latest_version['version']}")
            print(f"    Subject: {latest_version.get('subject_code', 'unknown')}")
            print(f"    Objects: {latest_version.get('num_objects', 'unknown')}")
            print(f"    Created: {latest_version.get('created_at', 'unknown')}")
        else:
            print("\n  No versions found in dataset path.")
    else:
        print("\n  No dataset directory found.")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="AION Trainer — Academic Foundation Model Training Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Parse documents and build dataset")
    ingest_parser.add_argument("--subject", required=True, help="Subject code (e.g., BAI401)")
    ingest_parser.add_argument("--books", help="Comma-separated paths to textbooks")
    ingest_parser.add_argument("--papers", help="Comma-separated paths to previous papers")
    ingest_parser.add_argument("--notes", help="Comma-separated paths to notes")
    ingest_parser.add_argument("--output", default="dataset/", help="Output directory")
    ingest_parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    ingest_parser.add_argument("--batch-size", type=int, default=50, help="Pages per batch")

    # Train
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config", default="configs/train.yaml", help="Training config")
    train_parser.add_argument("--dataset", help="Override dataset path")
    train_parser.add_argument("--epochs", type=int, help="Override number of epochs")
    train_parser.add_argument("--batch-size", type=int, help="Override batch size")
    train_parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    train_parser.add_argument("--resume", help="Resume from checkpoint")

    # Benchmark
    benchmark_parser = subparsers.add_parser("benchmark", help="Evaluate model")
    benchmark_parser.add_argument("--config", help="Training config")
    benchmark_parser.add_argument("--checkpoint", help="Specific checkpoint to evaluate")
    benchmark_parser.add_argument("--subject", help="Subject code")

    # Export
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument("--config", help="Training config")
    export_parser.add_argument("--checkpoint", help="Checkpoint to export")
    export_parser.add_argument("--output", default="outputs/", help="Output directory")
    export_parser.add_argument("--format", default="onnx", choices=["onnx", "pt", "safetensors"])

    # Continual
    continual_parser = subparsers.add_parser("continual", help="Incremental training")
    continual_parser.add_argument("--config", default="configs/train.yaml", help="Training config")
    continual_parser.add_argument("--dataset", help="New dataset path")

    # Curriculum
    curriculum_parser = subparsers.add_parser("curriculum", help="Show curriculum schedule")
    curriculum_parser.add_argument("--config", default="configs/train.yaml", help="Training config")

    # Status
    status_parser = subparsers.add_parser("status", help="Show training status")
    status_parser.add_argument("--config", default="configs/train.yaml", help="Training config")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Route to command handler
    commands = {
        "ingest": cmd_ingest,
        "train": cmd_train,
        "benchmark": cmd_benchmark,
        "export": cmd_export,
        "continual": cmd_continual,
        "curriculum": cmd_curriculum,
        "status": cmd_status,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
