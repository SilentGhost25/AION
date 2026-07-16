import threading
import pytest

from server.job_queue import JobQueue


def test_submit_creates_queued_job(job_queue):
    job = job_queue.submit("BAI401", "learn", params={"force": False})
    assert job.status == "queued"
    assert job.subject == "BAI401"
    assert job.id.startswith("JOB-")


def test_get_returns_none_for_unknown_job(job_queue):
    assert job_queue.get("JOB-DOES-NOT-EXIST") is None


def test_list_filters_by_subject_and_status(job_queue):
    job_queue.submit("BAI401", "learn")
    job_queue.submit("BCS402", "learn")
    job1 = job_queue.submit("BAI401", "benchmark", resource="cpu")

    job_queue.update_status(job1.id, "completed")

    bai_jobs = job_queue.list(subject="BAI401")
    assert len(bai_jobs) == 2

    completed = job_queue.list(status="completed")
    assert len(completed) == 1
    assert completed[0].id == job1.id


def test_claim_next_returns_none_when_empty(job_queue):
    assert job_queue.claim_next(resource="gpu") is None


def test_claim_next_marks_job_running(job_queue):
    submitted = job_queue.submit("BAI401", "learn", resource="gpu")
    claimed = job_queue.claim_next(resource="gpu")

    assert claimed.id == submitted.id
    assert claimed.status == "running"
    assert claimed.started_at is not None

    assert job_queue.claim_next(resource="gpu") is None


def test_claim_next_respects_resource_lane(job_queue):
    job_queue.submit("BAI401", "learn", resource="gpu")
    cpu_job = job_queue.submit("BAI401", "benchmark", resource="cpu")

    claimed = job_queue.claim_next(resource="cpu")
    assert claimed.id == cpu_job.id


def test_claim_next_is_fifo(job_queue):
    first = job_queue.submit("BAI401", "learn", resource="gpu")
    second = job_queue.submit("BCS402", "learn", resource="gpu")

    claimed = job_queue.claim_next(resource="gpu")
    assert claimed.id == first.id
    claimed2 = job_queue.claim_next(resource="gpu")
    assert claimed2.id == second.id


def test_no_job_claimed_twice_under_concurrent_threads(job_queue):
    num_jobs = 20
    num_threads = 8
    for i in range(num_jobs):
        job_queue.submit(f"SUBJECT{i}", "learn", resource="gpu")

    claimed_ids = []
    lock = threading.Lock()

    def worker():
        while True:
            job = job_queue.claim_next(resource="gpu")
            if job is None:
                return
            with lock:
                claimed_ids.append(job.id)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(claimed_ids) == num_jobs
    assert len(set(claimed_ids)) == num_jobs


def test_update_status_sets_finished_at_on_terminal_state(job_queue):
    job = job_queue.submit("BAI401", "learn")
    job_queue.update_status(job.id, "completed", result={"ok": True})

    fetched = job_queue.get(job.id)
    assert fetched.status == "completed"
    assert fetched.finished_at is not None
    assert fetched.result == {"ok": True}


def test_update_status_records_error_on_failure(job_queue):
    job = job_queue.submit("BAI401", "learn")
    job_queue.update_status(job.id, "failed", error="boom")

    fetched = job_queue.get(job.id)
    assert fetched.status == "failed"
    assert fetched.error == "boom"


def test_append_and_get_logs_in_order(job_queue):
    job = job_queue.submit("BAI401", "learn")
    job_queue.append_log(job.id, "first message")
    job_queue.append_log(job.id, "second message", level="WARNING")

    logs = job_queue.get_logs(job.id)
    assert len(logs) == 2
    assert logs[0]["message"] == "first message"
    assert logs[1]["level"] == "WARNING"


def test_get_logs_since_seq_returns_only_new_entries(job_queue):
    job = job_queue.submit("BAI401", "learn")
    job_queue.append_log(job.id, "one")
    job_queue.append_log(job.id, "two")

    all_logs = job_queue.get_logs(job.id)
    since = job_queue.get_logs(job.id, since_seq=all_logs[0]["seq"])
    assert len(since) == 1
    assert since[0]["message"] == "two"


def test_stream_logs_yields_terminal_marker_when_job_completes(job_queue):
    job = job_queue.submit("BAI401", "learn")
    job_queue.append_log(job.id, "working...")
    job_queue.update_status(job.id, "completed")

    entries = list(job_queue.stream_logs(job.id, poll_interval=0.01))
    assert any(e.get("terminal") for e in entries)
    assert entries[0]["message"] == "working..."


def test_stream_logs_returns_immediately_for_unknown_job(job_queue):
    entries = list(job_queue.stream_logs("JOB-NOT-REAL", poll_interval=0.01))
    assert entries == []
