# AION-Trainer/server/pipeline_runner.py
"""
Pipeline Runner — executes the full learn lifecycle for one job.
"""

import os
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

from server.manifest import ManifestManager, SubjectManifest
from server.model_registry import ModelRegistry, ModelRecord
from server.job_queue import JobQueue, Job

from checkpoints.manager import CheckpointManager
from benchmarks.evaluator import BenchmarkEvaluator

logger = logging.getLogger("aion")


class JobLogHandler(logging.Handler):
    def __init__(self, job_queue: JobQueue, job_id: str):
        super().__init__()
        self.job_queue = job_queue
        self.job_id = job_id

    def emit(self, record: logging.LogRecord):
        try:
            metrics = getattr(record, "metrics", None)
            self.job_queue.append_log(
                self.job_id, self.format(record), level=record.levelname, metrics=metrics,
            )
        except Exception:
            pass


def _default_parser_factory(subject: str, config: Dict[str, Any]):
    from preprocessing.parallel_parser import ParallelParser
    return ParallelParser(
        num_workers=config["server"].get("parse_workers", 8),
        batch_size=config["server"].get("parse_batch_size", 50),
        subject_code=subject,
    )


def _default_dataset_builder_factory(subject: str, config: Dict[str, Any]):
    from dataset.builder import DatasetBuilder
    return DatasetBuilder(output_dir=config["server"]["dataset_root"], subject_code=subject)


def _default_trainer_factory(train_config: Dict[str, Any]):
    from trainer.aion_trainer import AIONTrainer
    return AIONTrainer(train_config)


def _default_examiner_scorer_factory(candidate_weights_path: str):
    from server.candidate_generator import TransformersCandidateGenerator, NullCandidateGenerator
    from server.examiner_similarity import ExaminerSimilarityScorer
    try:
        generator = TransformersCandidateGenerator(candidate_weights_path)
    except Exception as e:
        logger.warning(f"[Pipeline] Could not load candidate model for ESS "
                        f"({e}); examiner similarity will be 0.0 this run.")
        generator = NullCandidateGenerator()
    return ExaminerSimilarityScorer(generator)


class PipelineRunner:
    def __init__(
        self,
        config: Dict[str, Any],
        job_queue: JobQueue,
        parser_factory: Callable = _default_parser_factory,
        dataset_builder_factory: Callable = _default_dataset_builder_factory,
        trainer_factory: Callable = _default_trainer_factory,
        examiner_scorer_factory: Callable = _default_examiner_scorer_factory,
        checkpoint_manager: Optional[CheckpointManager] = None,
        benchmark_evaluator: Optional[BenchmarkEvaluator] = None,
        manifest_manager: Optional[ManifestManager] = None,
        model_registry: Optional[ModelRegistry] = None,
    ):
        self.config = config
        self.job_queue = job_queue
        self.academic_root = config["server"]["academic_root"]
        self.dataset_root = config["server"]["dataset_root"]
        self.knowledge_root = config["server"]["knowledge_root"]

        self._parser_factory = parser_factory
        self._dataset_builder_factory = dataset_builder_factory
        self._trainer_factory = trainer_factory
        self._examiner_scorer_factory = examiner_scorer_factory

        self.manifest_manager = manifest_manager or ManifestManager(self.academic_root)
        self.model_registry = model_registry or ModelRegistry(config["server"]["models_root"])
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(config["checkpoints"]["dir"])
        self.benchmark_evaluator = benchmark_evaluator or BenchmarkEvaluator(config.get("benchmark", {}))

    def run(self, job: Job):
        handler = JobLogHandler(self.job_queue, job.id)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        try:
            if job.job_type == "learn":
                result = self._run_learn(job)
            elif job.job_type == "benchmark":
                result = self._run_benchmark(job)
            elif job.job_type == "evaluate":
                result = self._run_evaluate(job)
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")

            self.job_queue.update_status(job.id, "completed", result=result)

        except Exception as e:
            logger.error(f"[Pipeline] Job {job.id} failed: {e}")
            self.job_queue.update_status(job.id, "failed", error=str(e))
        finally:
            logger.removeHandler(handler)

    # ------------------------------------------------------------------
    # LEARN
    # ------------------------------------------------------------------

    def _run_learn(self, job: Job) -> Dict[str, Any]:
        subject = job.subject
        force = job.params.get("force", False)

        logger.info(f"[Pipeline] === Learning cycle started for {subject} ===")

        logger.info("[Pipeline] Stage 1/8: Discovering files...")
        old_manifest, new_manifest, diff = self.manifest_manager.get_or_create(subject)

        if not diff.has_changes and not force:
            logger.info("[Pipeline] No new or modified material found. Nothing to learn.")
            return {"status": "up_to_date", "message": "No changes detected since last learn."}

        logger.info(f"[Pipeline] Changes detected: +{len(diff.added)} added, "
                    f"~{len(diff.modified)} modified, -{len(diff.removed)} removed")

        logger.info("[Pipeline] Stage 2/8: Validating files...")
        changed_files = [str(Path(self.academic_root) / e.path) for e in (diff.added + diff.modified)]
        changed_files = [f for f in changed_files if Path(f).exists()]
        if not changed_files and not force:
            logger.warning("[Pipeline] Changed files listed but none found on disk. Aborting.")
            return {"status": "error", "message": "Manifest/disk mismatch."}

        logger.info(f"[Pipeline] Stage 3/8: Parsing {len(changed_files)} changed files...")
        parser = self._parser_factory(subject, self.config)
        new_objects = parser.parse_all(changed_files) if changed_files else []
        logger.info(f"[Pipeline] Extracted {len(new_objects)} new Knowledge Objects")

        logger.info("[Pipeline] Stage 4/8: Updating Knowledge Store...")
        all_objects = self._merge_knowledge_store(subject, new_objects)
        knowledge_version = self._bump_knowledge_version(subject)
        logger.info(f"[Pipeline] Knowledge store now has {len(all_objects)} objects "
                    f"(knowledge v{knowledge_version})")

        logger.info("[Pipeline] Stage 5/8: Building training dataset...")
        builder = self._dataset_builder_factory(subject, self.config)
        builder.build(all_objects)

        from dataset.version import DatasetVersion
        version_mgr = DatasetVersion(self.dataset_root)
        version_info = version_mgr.create_version(
            subject_code=subject,
            source_files=[e.path for e in (diff.added + diff.modified)],
            num_objects=len(all_objects),
        )
        dataset_version = version_info["version"]
        logger.info(f"[Pipeline] Dataset version {dataset_version} created")

        logger.info("[Pipeline] Stage 6/8: Training...")
        train_config = self._build_train_config(subject, dataset_version)
        trainer = self._trainer_factory(train_config)
        trainer.train()
        latest_checkpoint = self.checkpoint_manager.get_latest()

        if latest_checkpoint is None:
            raise RuntimeError("Training completed but no checkpoint was produced.")

        logger.info("[Pipeline] Stage 7/8: Benchmarking candidate...")
        benchmark_scores = self.benchmark_evaluator.evaluate(latest_checkpoint, subject_code=subject)

        ess_scores = self._compute_examiner_similarity(subject, latest_checkpoint, dataset_version)
        benchmark_scores.update(ess_scores)
        logger.info(f"[Pipeline] Benchmark overall score: {benchmark_scores.get('overall_score', 0):.4f}",
                    extra={"metrics": benchmark_scores})

        logger.info("[Pipeline] Stage 8/8: Registering candidate...")
        record = self.model_registry.register_candidate(
            subject=subject, weights_path=latest_checkpoint,
            dataset_version=dataset_version, knowledge_version=knowledge_version,
            benchmark_scores=benchmark_scores, job_id=job.id,
        )

        new_manifest.last_trained = job.created_at
        new_manifest.dataset_version = dataset_version
        self.manifest_manager.save(new_manifest)

        gate = self.model_registry.compare_to_production(subject, benchmark_scores)
        logger.info(f"[Pipeline] === Learning cycle complete: candidate AION_{record.version} "
                    f"({'beats' if gate['can_promote'] else 'does not beat'} production) ===")

        return {
            "status": "completed", "candidate_version": record.version,
            "dataset_version": dataset_version, "knowledge_version": knowledge_version,
            "benchmark_scores": benchmark_scores, "can_promote": gate["can_promote"],
            "comparisons": gate["comparisons"],
        }

    def _compute_examiner_similarity(
        self, subject: str, checkpoint_path: str, dataset_version: str
    ) -> Dict[str, float]:
        try:
            manifest = self.manifest_manager.load_manifest(subject)
            previous_paper_paths = [
                str(Path(self.academic_root) / e.path)
                for e in (manifest.previous_papers if manifest else [])
            ]
            knowledge_samples = self._sample_knowledge_texts(subject, dataset_version)

            scorer = self._examiner_scorer_factory(checkpoint_path)
            return scorer.compute(previous_paper_paths, knowledge_samples)
        except Exception as e:
            logger.warning(f"[Pipeline] Examiner similarity computation failed: {e}")
            return {"examiner_similarity_score": 0.0}

    def _sample_knowledge_texts(self, subject: str, dataset_version: str, limit: int = 50) -> List[str]:
        dataset_file = Path(self.dataset_root) / subject / dataset_version / "train.jsonl"
        if not dataset_file.exists():
            dataset_file = Path(self.dataset_root) / subject / "train.jsonl"
        if not dataset_file.exists():
            return []

        texts = []
        with open(dataset_file) as f:
            for line in f:
                if len(texts) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                if sample.get("knowledge"):
                    texts.append(sample["knowledge"])
        return texts

    def _merge_knowledge_store(self, subject: str, new_objects) -> list:
        from preprocessing.parallel_parser import KnowledgeObject

        store_dir = Path(self.knowledge_root) / subject
        store_dir.mkdir(parents=True, exist_ok=True)
        store_file = store_dir / "objects.jsonl"

        existing = []
        if store_file.exists():
            with open(store_file) as f:
                existing = [json.loads(line) for line in f if line.strip()]

        existing_ids = {o["object_id"] for o in existing}
        for obj in new_objects:
            if obj.object_id not in existing_ids:
                existing.append(obj.to_dict())

        with open(store_file, "w") as f:
            for obj in existing:
                f.write(json.dumps(obj) + "\n")

        return [KnowledgeObject(**o) for o in existing]

    def _bump_knowledge_version(self, subject: str) -> str:
        version_file = Path(self.knowledge_root) / subject / "version.txt"
        current = 0.0
        if version_file.exists():
            current = float(version_file.read_text().strip() or "0.0")
        new_version = round(current + 0.1, 1)
        version_file.parent.mkdir(parents=True, exist_ok=True)
        version_file.write_text(str(new_version))
        return str(new_version)

    def _build_train_config(self, subject: str, dataset_version: str) -> Dict[str, Any]:
        base_config_path = self.config["server"].get("train_config", "configs/train.yaml")
        if not os.path.exists(base_config_path):
            # Create a simple yaml for mock runner if it doesn't exist
            os.makedirs(os.path.dirname(base_config_path), exist_ok=True)
            with open(base_config_path, "w") as f:
                yaml.dump({"dataset": {"path": ""}, "training": {"epochs": 10}}, f)
        with open(base_config_path) as f:
            train_config = yaml.safe_load(f)
        train_config["dataset"]["path"] = str(Path(self.dataset_root) / subject / dataset_version)
        return train_config

    # ------------------------------------------------------------------
    # BENCHMARK / EVALUATE
    # ------------------------------------------------------------------

    def _run_run_benchmark(self, job: Job) -> Dict[str, Any]:
        subject = job.subject
        production = self.model_registry.get_production(subject)
        if production is None:
            return {"status": "error", "message": "No production model to benchmark."}

        logger.info(f"[Pipeline] Benchmarking production model AION_{production.version}...")
        scores = self.benchmark_evaluator.evaluate(production.weights_path, subject_code=subject)
        logger.info("[Pipeline] Benchmark complete.", extra={"metrics": scores})
        return {"status": "completed", "version": production.version, "scores": scores}

    def _run_run_evaluate(self, job: Job) -> Dict[str, Any]:
        subject = job.subject
        candidate = self.model_registry.get_candidate(subject, job.params.get("version"))
        if candidate is None:
            return {"status": "error", "message": "No candidate to evaluate."}

        logger.info(f"[Pipeline] Deep-evaluating candidate AION_{candidate.version}...")
        scores = dict(candidate.benchmark_scores)
        ess_scores = self._compute_examiner_similarity(
            subject, candidate.weights_path, candidate.dataset_version
        )
        scores.update(ess_scores)
        logger.info(f"[Pipeline] Examiner Similarity Score: "
                    f"{scores.get('examiner_similarity_score', 0):.4f}", extra={"metrics": scores})
        return {"status": "completed", "version": candidate.version, "scores": scores}

    def _run_benchmark(self, job: Job) -> Dict[str, Any]:
        return self._run_run_benchmark(job)

    def _run_evaluate(self, job: Job) -> Dict[str, Any]:
        return self._run_run_evaluate(job)
