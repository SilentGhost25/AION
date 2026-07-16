# AION-Trainer/server/model_registry.py
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

class ModelRecord:
    def __init__(
        self,
        version: str,
        subject: str,
        weights_path: str,
        dataset_version: str,
        knowledge_version: str,
        benchmark_scores: Dict[str, float],
        job_id: str,
        registered_at: Optional[str] = None,
        is_production: bool = False,
    ):
        self.version = version
        self.subject = subject
        self.weights_path = weights_path
        self.dataset_version = dataset_version
        self.knowledge_version = knowledge_version
        self.benchmark_scores = benchmark_scores
        self.job_id = job_id
        self.registered_at = registered_at or datetime.utcnow().isoformat()
        self.is_production = is_production

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "subject": self.subject,
            "weights_path": self.weights_path,
            "dataset_version": self.dataset_version,
            "knowledge_version": self.knowledge_version,
            "benchmark_scores": self.benchmark_scores,
            "job_id": self.job_id,
            "registered_at": self.registered_at,
            "is_production": self.is_production,
        }

class ModelRegistry:
    def __init__(self, models_root: str):
        self.models_root = models_root
        os.makedirs(models_root, exist_ok=True)
        self.registry_file = os.path.join(models_root, "registry.json")
        self._load()

    def _load(self):
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, encoding="utf-8") as f:
                    data = json.load(f)
                    self.records = [ModelRecord(**r) for r in data]
            except Exception:
                self.records = []
        else:
            self.records = []

    def _save(self):
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.records], f, indent=2)

    def register_candidate(
        self,
        subject: str,
        weights_path: str,
        dataset_version: str,
        knowledge_version: str,
        benchmark_scores: Dict[str, float],
        job_id: str,
    ) -> ModelRecord:
        subject_records = [r for r in self.records if r.subject == subject]
        
        if not subject_records:
            next_version = "0.1"
        else:
            versions = []
            for r in subject_records:
                try:
                    versions.append(float(r.version))
                except ValueError:
                    pass
            next_version = f"{round(max(versions) + 0.1, 1)}" if versions else "0.1"

        record = ModelRecord(
            version=next_version,
            subject=subject,
            weights_path=weights_path,
            dataset_version=dataset_version,
            knowledge_version=knowledge_version,
            benchmark_scores=benchmark_scores,
            job_id=job_id,
            is_production=False,
        )
        self.records.append(record)
        self._save()
        return record

    def get_production(self, subject: str) -> Optional[ModelRecord]:
        for r in self.records:
            if r.subject == subject and r.is_production:
                return r
        return None

    def get_candidate(self, subject: str, version: str) -> Optional[ModelRecord]:
        for r in self.records:
            if r.subject == subject and r.version == version:
                return r
        return None

    def list_candidates(self, subject: str) -> List[ModelRecord]:
        return [r for r in self.records if r.subject == subject]

    def promote_to_production(self, subject: str, version: str) -> bool:
        target = self.get_candidate(subject, version)
        if not target:
            return False
        for r in self.records:
            if r.subject == subject:
                r.is_production = (r.version == version)
        self._save()
        return True

    def compare_to_production(self, subject: str, candidate_scores: Dict[str, float]) -> Dict[str, Any]:
        production = self.get_production(subject)
        if not production:
            return {"can_promote": True, "comparisons": {}}

        comparisons = {}
        can_promote = True
        
        all_metrics = set(production.benchmark_scores.keys()) | set(candidate_scores.keys())
        for m in all_metrics:
            prod_val = production.benchmark_scores.get(m, 0.0)
            cand_val = candidate_scores.get(m, 0.0)
            diff = cand_val - prod_val
            comparisons[m] = {"current": prod_val, "candidate": cand_val, "diff": diff}
            if m == "overall_score" and diff < 0:
                can_promote = False

        return {"can_promote": can_promote, "comparisons": comparisons}
