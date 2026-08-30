"""tests/test_jobs_redis.py — Redis-backed job persistence for the job queue."""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path

import fakeredis
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.jobs import JobManager, JobStatus


def _wait_status(manager: JobManager, job_id: str, timeout: float = 10.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = manager.get(job_id)
        if j and j.status in (JobStatus.SUCCESS, JobStatus.FAILED):
            return j.status.value
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish")


def test_redis_backed_job_persists_state():
    r = fakeredis.FakeStrictRedis()
    m = JobManager(redis_client=r)

    job = m.create("test_task")
    m.submit(job, lambda j: {"answer": 42})

    assert _wait_status(m, job.id) == "success"
    j = m.get(job.id)
    assert j.result == {"answer": 42}
    assert j.status == JobStatus.SUCCESS

    # state must be visible in redis directly
    raw = r.hgetall(f"cell_analysis:job:{job.id}")
    assert raw, "job hash missing in redis"
    data = json.loads(raw[b"data"])
    assert data["status"] == "success"
    assert data["result"] == {"answer": 42}


def test_redis_backed_job_survives_manager_restart():
    """A new JobManager on the same redis must see finished jobs (restart)."""
    r = fakeredis.FakeStrictRedis()
    m1 = JobManager(redis_client=r)
    job = m1.create("restart_task")
    m1.submit(job, lambda j: {"done": True})
    assert _wait_status(m1, job.id) == "success"

    m2 = JobManager(redis_client=r)  # simulates server restart
    j2 = m2.get(job.id)
    assert j2 is not None
    assert j2.status == JobStatus.SUCCESS
    assert j2.result == {"done": True}
    assert j2.log, "log not restored"


def test_redis_backed_list_jobs():
    r = fakeredis.FakeStrictRedis()
    m = JobManager(redis_client=r)
    j1 = m.create("a")
    m.submit(j1, lambda j: {})
    j2 = m.create("b")
    m.submit(j2, lambda j: {})
    _wait_status(m, j1.id)
    _wait_status(m, j2.id)

    m2 = JobManager(redis_client=r)
    jobs = m2.list_jobs(limit=10)
    ids = {j["id"] for j in jobs}
    assert {j1.id, j2.id} <= ids


def test_in_memory_fallback_without_redis():
    """No redis client -> plain in-memory behaviour (existing tests rely on it)."""
    m = JobManager()  # no redis
    job = m.create("mem")
    m.submit(job, lambda j: {"x": 1})
    assert _wait_status(m, job.id) == "success"
    assert m.get(job.id).result == {"x": 1}
