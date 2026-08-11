"""
src/api/jobs.py — simple in-memory async job queue for long-running pipeline tasks.
Suitable for HF Spaces where request timeouts are short.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


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


class JobManager:
    def __init__(self, max_workers: int = 2, max_jobs: int = 200):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._max_workers = max_workers
        self._max_jobs = max_jobs
        self._semaphore = threading.Semaphore(max_workers)

    def create(self, task: str) -> Job:
        job = Job(id=str(uuid.uuid4()), task=task)
        with self._lock:
            self._jobs[job.id] = job
            self._prune_locked()
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

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [j.to_dict() for j in jobs[:limit]]

    def _append_log(self, job: Job, message: str) -> None:
        with self._lock:
            job.log.append(message)

    def _set_status(self, job: Job, status: JobStatus) -> None:
        with self._lock:
            job.status = status
            if status == JobStatus.RUNNING:
                job.started_at = time.time()
            if status in (JobStatus.SUCCESS, JobStatus.FAILED):
                job.finished_at = time.time()

    def _set_result(self, job: Job, result: Dict[str, Any]) -> None:
        with self._lock:
            job.result = result

    def _set_error(self, job: Job, error: str) -> None:
        with self._lock:
            job.error = error

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
