"""
AION Academic Reasoning Dataset v1.0 (ARD v1) Exporter
Supports local workspace and Google Colab / Google Drive execution.

Features:
- Schema validation via jsonschema (with graceful fallback)
- Automatic train/val/test splitting (80/10/10 default)
- Multi-task JSONL export (train_by_task & train_by_subject)
- Manifest generation with sha256 checksums & dataset stats
"""

import os
import sys
import json
import glob
import hashlib
import random
from pathlib import Path
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


def compute_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def export_ard_v1(
    base_dir: str = ".",
    export_dir: str = "exports",
    min_critic_score: float = 0.85,
    exclude_flagged: bool = True,
    exclude_pending: bool = True,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    base_path = Path(base_dir).resolve()
    export_path = base_path / export_dir
    schema_file = base_path / "datasets" / "schema" / "ard_v1_schema.json"
    samples_dir = base_path / "datasets" / "samples"

    print(f"🚀 Initializing ARD v1 Exporter...")
    print(f"   Root Directory: {base_path}")
    print(f"   Export Directory: {export_path}")

    # Create directories
    (export_path / "train_by_task").mkdir(parents=True, exist_ok=True)
    (export_path / "train_by_subject").mkdir(parents=True, exist_ok=True)
    (base_path / "datasets" / "index").mkdir(parents=True, exist_ok=True)

    # Load Schema if available
    schema = None
    if schema_file.exists():
        with open(schema_file, "r", encoding="utf-8") as f:
            schema = json.load(f)
        print(f"   Loaded Schema: {schema_file.name}")

    # Collect samples
    sample_files = list(samples_dir.glob("**/*.json"))
    print(f"   Found {len(sample_files)} sample files in {samples_dir}")

    processed_samples = []
    task_counts = {}
    subject_counts = {}
    bloom_counts = {"L1": 0, "L2": 0, "L3": 0, "L4": 0, "L5": 0, "L6": 0}
    exam_counts = {}
    dept_counts = {}
    review_status_counts = {"approved": 0, "pending": 0, "rejected": 0, "flagged": 0}
    critic_scores = []
    negative_count = 0

    for s_file in sample_files:
        try:
            with open(s_file, "r", encoding="utf-8") as f:
                sample = json.load(f)

            # Validate against schema if available
            if HAS_JSONSCHEMA and schema:
                jsonschema.validate(instance=sample, schema=schema)

            prov = sample.get("provenance", {})
            meta = sample.get("metadata", {})
            critic = sample.get("critic", {})
            neg = sample.get("negative")

            score = critic.get("overall_score", 0.0)
            status = prov.get("review_status", "pending")
            review_status_counts[status] = review_status_counts.get(status, 0) + 1

            if exclude_flagged and status == "flagged":
                continue
            if exclude_pending and status == "pending":
                continue
            if score < min_critic_score and status != "approved":
                continue

            processed_samples.append(sample)
            critic_scores.append(score)

            t_type = prov.get("task_type", "UNKNOWN")
            task_counts[t_type] = task_counts.get(t_type, 0) + 1

            subj = meta.get("subject", "UNKNOWN")
            subject_counts[subj] = subject_counts.get(subj, 0) + 1

            bloom = meta.get("bloom_level", "L1")
            if bloom in bloom_counts:
                bloom_counts[bloom] += 1

            exam = meta.get("exam_type", "IA")
            exam_counts[exam] = exam_counts.get(exam, 0) + 1

            dept = meta.get("department", "ECE")
            dept_counts[dept] = dept_counts.get(dept, 0) + 1

            if neg and neg.get("is_negative_sample"):
                negative_count += 1

        except Exception as e:
            print(f"⚠️ Warning: Skipped {s_file.name} — {e}")

    print(f"   Valid & Filtered Samples: {len(processed_samples)}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(processed_samples)

    n_total = len(processed_samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_samples = processed_samples[:n_train]
    val_samples = processed_samples[n_train:n_train + n_val]
    test_samples = processed_samples[n_train + n_val:]

    def write_jsonl(filepath: Path, samples: list):
        with open(filepath, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Export main splits
    train_file = export_path / "train.jsonl"
    val_file = export_path / "val.jsonl"
    test_file = export_path / "test.jsonl"

    write_jsonl(train_file, train_samples)
    write_jsonl(val_file, val_samples)
    write_jsonl(test_file, test_samples)

    # Export by task
    by_task = {}
    for s in processed_samples:
        t = s["provenance"]["task_type"]
        by_task.setdefault(t, []).append(s)

    for task_name, t_samples in by_task.items():
        write_jsonl(export_path / "train_by_task" / f"{task_name}.jsonl", t_samples)

    # Export by subject
    by_subject = {}
    for s in processed_samples:
        subj_code = s["metadata"].get("subject_code", "GENERIC")
        by_subject.setdefault(subj_code, []).append(s)

    for subj_code, s_samples in by_subject.items():
        write_jsonl(export_path / "train_by_subject" / f"{subj_code}.jsonl", s_samples)

    # Generate Manifest
    avg_score = sum(critic_scores) / len(critic_scores) if critic_scores else 0.0

    manifest = {
        "dataset_name": "AION Academic Reasoning Dataset",
        "version": "1.0",
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "statistics": {
            "total_samples": n_total,
            "by_task_type": task_counts,
            "by_exam_type": exam_counts,
            "by_department": dept_counts,
            "by_bloom_level": bloom_counts,
            "by_review_status": review_status_counts,
            "average_critic_score": round(avg_score, 4),
            "negative_samples": negative_count
        },
        "splits": {
            "train": {"samples": len(train_samples), "path": "exports/train.jsonl"},
            "val": {"samples": len(val_samples), "path": "exports/val.jsonl"},
            "test": {"samples": len(test_samples), "path": "exports/test.jsonl"}
        },
        "checksums": {
            "train.jsonl": compute_sha256(train_file) if train_file.exists() and n_total > 0 else "",
            "val.jsonl": compute_sha256(val_file) if val_file.exists() and n_total > 0 else "",
            "test.jsonl": compute_sha256(test_file) if test_file.exists() and n_total > 0 else ""
        },
        "filtering_applied": {
            "min_critic_score": min_critic_score,
            "exclude_flagged": exclude_flagged,
            "exclude_pending": exclude_pending
        }
    }

    manifest_file = base_path / "datasets" / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Export completed successfully!")
    print(f"   Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    print(f"   Manifest saved to: {manifest_file}")


if __name__ == "__main__":
    export_ard_v1()
