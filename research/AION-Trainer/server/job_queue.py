# AION-Trainer/server/job_queue.py
import sqlite3
import json
import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import threading

class Job:
    def __init__(
        self,
        id: str,
        subject: str,
        job_type: str,
        status: str,
        resource: str = "gpu",
        params: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        self.id = id
        self.subject = subject
        self.job_type = job_type
        self.status = status
        self.resource = resource
        self.params = params or {}
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.started_at = started_at
        self.finished_at = finished_at
        self.result = result
        self.error = error

class JobQueue:
    def __init__(self, db_path: str = "jobs.db"):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                        id TEXT PRIMARY KEY,
                        subject TEXT,
                        job_type TEXT,
                        status TEXT,
                        resource TEXT,
                        params TEXT,
                        created_at TEXT,
                        started_at TEXT,
                        finished_at TEXT,
                        result TEXT,
                        error TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS logs (
                        seq INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT,
                        message TEXT,
                        level TEXT,
                        timestamp TEXT,
                        metrics TEXT
                    )
                    """
                )
                conn.commit()

    def submit(self, subject: str, job_type: str, resource: str = "gpu", params: Optional[Dict[str, Any]] = None) -> Job:
        with self._lock:
            job_id = f"JOB-{uuid.uuid4().hex[:8].upper()}"
            job = Job(
                id=job_id,
                subject=subject,
                job_type=job_type,
                status="queued",
                resource=resource,
                params=params,
            )
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO jobs (id, subject, job_type, status, resource, params, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (job.id, job.subject, job.job_type, job.status, job.resource, json.dumps(job.params), job.created_at),
                )
                conn.commit()
            return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, subject, job_type, status, resource, params, created_at, started_at, finished_at, result, error FROM jobs WHERE id = ?", (job_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                return Job(
                    id=row[0],
                    subject=row[1],
                    job_type=row[2],
                    status=row[3],
                    resource=row[4],
                    params=json.loads(row[5]) if row[5] else {},
                    created_at=row[6],
                    started_at=row[7],
                    finished_at=row[8],
                    result=json.loads(row[9]) if row[9] else None,
                    error=row[10],
                )

    def list(self, subject: Optional[str] = None, status: Optional[str] = None) -> List[Job]:
        with self._lock:
            query = "SELECT id, subject, job_type, status, resource, params, created_at, started_at, finished_at, result, error FROM jobs WHERE 1=1"
            params = []
            if subject:
                query += " AND subject = ?"
                params.append(subject)
            if status:
                query += " AND status = ?"
                params.append(status)
            query += " ORDER BY created_at ASC"
            
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                jobs = []
                for row in rows:
                    jobs.append(
                        Job(
                            id=row[0],
                            subject=row[1],
                            job_type=row[2],
                            status=row[3],
                            resource=row[4],
                            params=json.loads(row[5]) if row[5] else {},
                            created_at=row[6],
                            started_at=row[7],
                            finished_at=row[8],
                            result=json.loads(row[9]) if row[9] else None,
                            error=row[10],
                        )
                    )
                return jobs

    def claim_next(self, resource: str = "gpu") -> Optional[Job]:
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM jobs WHERE status = 'queued' AND resource = ? ORDER BY created_at ASC LIMIT 1",
                    (resource,),
                )
                row = cursor.fetchone()
                if not row:
                    conn.rollback()
                    return None
                job_id = row[0]
                started_at = datetime.utcnow().isoformat()
                conn.execute(
                    "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
                    (started_at, job_id),
                )
                conn.commit()
            return self.get(job_id)

    def update_status(self, job_id: str, status: str, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        with self._lock:
            finished_at = datetime.utcnow().isoformat() if status in ("completed", "failed") else None
            with self._get_conn() as conn:
                if status in ("completed", "failed"):
                    conn.execute(
                        "UPDATE jobs SET status = ?, result = ?, error = ?, finished_at = ? WHERE id = ?",
                        (status, json.dumps(result) if result else None, error, finished_at, job_id),
                    )
                else:
                    conn.execute(
                        "UPDATE jobs SET status = ?, result = ?, error = ? WHERE id = ?",
                        (status, json.dumps(result) if result else None, error, job_id),
                    )
                conn.commit()

    def append_log(self, job_id: str, message: str, level: str = "INFO", metrics: Optional[Dict[str, Any]] = None):
        with self._lock:
            timestamp = datetime.utcnow().isoformat()
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO logs (job_id, message, level, timestamp, metrics) VALUES (?, ?, ?, ?, ?)",
                    (job_id, message, level, timestamp, json.dumps(metrics) if metrics else None),
                )
                conn.commit()

    def get_logs(self, job_id: str, since_seq: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            query = "SELECT seq, message, level, timestamp, metrics FROM logs WHERE job_id = ?"
            params = [job_id]
            if since_seq is not None:
                query += " AND seq > ?"
                params.append(since_seq)
            query += " ORDER BY seq ASC"
            
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                logs = []
                for row in rows:
                    logs.append(
                        {
                            "seq": row[0],
                            "message": row[1],
                            "level": row[2],
                            "timestamp": row[3],
                            "metrics": json.loads(row[4]) if row[4] else None,
                        }
                    )
                return logs

    def stream_logs(self, job_id: str, poll_interval: float = 0.5):
        """Yield log lines until the job completes or fails."""
        # First verify if the job exists
        job = self.get(job_id)
        if not job:
            return

        last_seq = -1
        while True:
            new_logs = self.get_logs(job_id, since_seq=last_seq)
            for log in new_logs:
                yield log
                last_seq = log["seq"]

            current_job = self.get(job_id)
            if current_job.status in ("completed", "failed"):
                # Fetch any remaining logs that might have been written during the final state update
                final_logs = self.get_logs(job_id, since_seq=last_seq)
                for log in final_logs:
                    yield log
                # Yield terminal marker
                yield {"terminal": True, "status": current_job.status}
                break

            time.sleep(poll_interval)
