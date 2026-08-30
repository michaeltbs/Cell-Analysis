"""
src/api/jobs.py — async job queue for long-running pipeline tasks.

In-memory by default (suitable for HF Spaces). When a redis client is
provided, job state is mirrored to redis hashes so jobs survive server
restarts (key prefix: cell_analysis:job:<id>).
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

REDIS_PREFIX = "cell_analysis:job:"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    task: str
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    log: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "log": self.log,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        return cls(
            id=data["id"],
            task=data.get("task", ""),
            status=JobStatus(data.get("status", "pending")),
            created_at=data.get("created_at", 0.0),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            result=data.get("result"),
            error=data.get("error"),
            log=list(data.get("log", [])),
        )


class JobManager:
    def __init__(self, max_workers: int = 2, max_jobs: int = 200, redis_client=None):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._max_workers = max_workers
        self._max_jobs = max_jobs
        self._semaphore = threading.Semaphore(max_workers)
        self._redis = redis_client

    # -- redis helpers -----------------------------------------------------
    def _redis_key(self, job_id: str) -> str:
        return f"{REDIS_PREFIX}{job_id}"

    def _redis_save(self, job: Job) -> None:
        if self._redis is None:
            return
        try:
            self._redis.hset(
                self._redis_key(job.id),
                mapping={
                    "data": json.dumps(job.to_dict(), default=str),
                },
            )
            self._redis.expire(self._redis_key(job.id), 86400)  # 24h TTL
        except Exception:
            pass  # redis is best-effort; in-memory still works

    def _redis_load(self, job_id: str) -> Optional[Job]:
        if self._redis is None:
            return None
        try:
            raw = self._redis.hget(self._redis_key(job_id), "data")
            if not raw:
                return None
            return Job.from_dict(json.loads(raw))
        except Exception:
            return None

    def _redis_delete(self, job_id: str) -> None:
        if self._redis is None:
            return
        try:
            self._redis.delete(self._redis_key(job_id))
        except Exception:
            pass

    def _redis_all_ids(self) -> List[str]:
        if self._redis is None:
            return []
        try:
            keys = self._redis.keys(f"{REDIS_PREFIX}*")
            ids = []
            for k in keys:
                k = k.decode() if isinstance(k, bytes) else k
                if k.startswith(REDIS_PREFIX):
                    ids.append(k[len(REDIS_PREFIX):])
            return ids
        except Exception:
            return []

    # -- core API ----------------------------------------------------------
    def create(self, task: str) -> Job:
        job = Job(id=str(uuid.uuid4()), task=task)
        with self._lock:
            self._jobs[job.id] = job
            self._prune_locked()
        self._redis_save(job)
        return job

    def _prune_locked(self) -> None:
        """Drop oldest finished jobs when over capacity to bound memory."""
        if len(self._jobs) <= self._max_jobs:
            return
        finished = sorted(
            (j for j in self._jobs.values() if j.status in (JobStatus.SUCCESS, JobStatus.FAILED)),
            key=lambda j: j.finished_at or 0,
        )
        overflow = len(self._jobs) - self._max_jobs
        for j in finished[:overflow]:
            self._jobs.pop(j.id, None)
            self._redis_delete(j.id)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job
        # not in memory (e.g. after restart) -> try redis
        return self._redis_load(job_id)

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            if len(jobs) < limit and self._redis is not None:
                # merge redis-only jobs (survived a restart)
                known = {j.id for j in jobs}
                for job_id in self._redis_all_ids():
                    if job_id in known:
                        continue
                    rj = self._redis_load(job_id)
                    if rj is not None:
                        jobs.append(rj)
                jobs.sort(key=lambda j: j.created_at, reverse=True)
            return [j.to_dict() for j in jobs[:limit]]

    def _append_log(self, job: Job, message: str) -> None:
        with self._lock:
            job.log.append(message)
        self._redis_save(job)

    def _set_status(self, job: Job, status: JobStatus) -> None:
        with self._lock:
            job.status = status
            if status == JobStatus.RUNNING:
                job.started_at = time.time()
            if status in (JobStatus.SUCCESS, JobStatus.FAILED):
                job.finished_at = time.time()
        self._redis_save(job)

    def _set_result(self, job: Job, result: Dict[str, Any]) -> None:
        with self._lock:
            job.result = result
        self._redis_save(job)

    def _set_error(self, job: Job, error: str) -> None:
        with self._lock:
            job.error = error
        self._redis_save(job)

    def submit(self, job: Job, fn: Callable[[Job], Dict[str, Any]]) -> None:
        def _run():
            with self._semaphore:
                self._set_status(job, JobStatus.RUNNING)
                self._append_log(job, "Job started")
                try:
                    result = fn(job)
                    self._set_result(job, result)
                    self._set_status(job, JobStatus.SUCCESS)
                    self._append_log(job, "Job finished")
                except Exception as e:
                    self._set_error(job, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
                    self._set_status(job, JobStatus.FAILED)
                    self._append_log(job, f"Job failed: {e}")

        threading.Thread(target=_run, daemon=True).start()


# global singleton
manager = JobManager()
