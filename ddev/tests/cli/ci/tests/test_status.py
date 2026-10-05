# (C) Datadog, Inc. 2026-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)
"""Tests for the status vocabulary: conclusion mapping and timeout classification."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ddev.cli.ci.tests.status import (
    TEST_JOB_TIMEOUT_MINUTES,
    Status,
    batch_status,
    conclusion_to_status,
    has_run,
    has_started_running,
    is_queued,
    job_status,
    timed_out,
)
from ddev.utils.github_async.models import WorkflowJob, WorkflowJobConclusion, WorkflowJobStatus
from tests.cli.ci.tests.helpers import RUNNER_NAME
from tests.helpers.github_async import make_workflow_job

REPO_ROOT = Path(__file__).parents[5]
TEST_BATCH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test-batch.yml"


def _cancelled_after(minutes: int, *, runner_name: str | None = RUNNER_NAME) -> WorkflowJob:
    """A job that ran on a runner for `minutes` and concluded `cancelled`."""
    return make_workflow_job(
        conclusion=WorkflowJobConclusion.CANCELLED,
        runner_name=runner_name,
        started_at="2026-01-01T10:00:00Z",
        completed_at=f"2026-01-01T{10 + minutes // 60:02d}:{minutes % 60:02d}:00Z",
    )


@pytest.mark.parametrize("job_id", ["test", "test_fork"])
def test_the_workflows_test_jobs_use_the_timeout_constant(job_id: str):
    """The constant classifies timeout stops, so the workflow's own limit must not drift from it."""
    workflow = yaml.safe_load(TEST_BATCH_WORKFLOW.read_text(encoding="utf-8"))
    assert workflow["jobs"][job_id]["timeout-minutes"] == TEST_JOB_TIMEOUT_MINUTES


@pytest.mark.parametrize(
    ("conclusion", "expected"),
    [
        ("success", Status.SUCCESS),
        ("skipped", Status.SKIPPED),
        ("failure", Status.FAILURE),
        ("cancelled", Status.CANCELLED),
        ("timed_out", Status.FAILURE),
        ("action_required", Status.FAILURE),
        ("neutral", Status.FAILURE),
        (None, Status.FAILURE),
    ],
)
def test_conclusion_to_status(conclusion: str | None, expected: Status):
    result = conclusion_to_status(conclusion)
    assert result is expected
    assert isinstance(result, Status)


@pytest.mark.parametrize(
    ("status", "conclusion", "runner_name", "queued", "started", "ran"),
    [
        pytest.param(WorkflowJobStatus.QUEUED, None, None, True, False, False, id="queued"),
        pytest.param(WorkflowJobStatus.WAITING, None, None, True, False, False, id="waiting"),
        pytest.param(WorkflowJobStatus.PENDING, None, None, True, False, False, id="pending"),
        pytest.param(WorkflowJobStatus.REQUESTED, None, None, False, False, False, id="requested"),
        pytest.param(WorkflowJobStatus.IN_PROGRESS, None, RUNNER_NAME, False, True, False, id="in-progress"),
        pytest.param(
            WorkflowJobStatus.COMPLETED, WorkflowJobConclusion.SUCCESS, RUNNER_NAME, False, True, True, id="success"
        ),
        pytest.param(
            WorkflowJobStatus.COMPLETED, WorkflowJobConclusion.FAILURE, RUNNER_NAME, False, True, True, id="failure"
        ),
        pytest.param(
            WorkflowJobStatus.COMPLETED,
            WorkflowJobConclusion.CANCELLED,
            RUNNER_NAME,
            False,
            True,
            True,
            id="ran-then-cancelled",
        ),
        # Skipped jobs and jobs cancelled while queued were never picked up by a runner.
        pytest.param(
            WorkflowJobStatus.COMPLETED, WorkflowJobConclusion.SKIPPED, None, False, False, False, id="skipped"
        ),
        pytest.param(
            WorkflowJobStatus.COMPLETED,
            WorkflowJobConclusion.CANCELLED,
            None,
            False,
            False,
            False,
            id="cancelled-while-queued",
        ),
    ],
)
def test_a_jobs_state_on_a_runner(
    status: WorkflowJobStatus,
    conclusion: WorkflowJobConclusion | None,
    runner_name: str | None,
    queued: bool,
    started: bool,
    ran: bool,
):
    """Only a job a runner was assigned to has timing worth measuring; `requested` is not a wait."""
    job = make_workflow_job(status=status, conclusion=conclusion, runner_name=runner_name)

    assert (is_queued(job), has_started_running(job), has_run(job)) == (queued, started, ran)


@pytest.mark.parametrize(
    "minutes", [TEST_JOB_TIMEOUT_MINUTES, TEST_JOB_TIMEOUT_MINUTES + 1], ids=["at-limit", "past-limit"]
)
def test_a_cancelled_job_that_ran_to_the_limit_timed_out(minutes: int):
    """GitHub reports a timeout stop as `cancelled`; running to the limit is what tells it apart."""
    job = _cancelled_after(minutes)
    assert timed_out(job)
    assert job_status(job) is Status.FAILURE


def test_a_cancelled_job_that_ran_short_of_the_limit_was_cancelled():
    """A shorter `cancelled` job was really cancelled: it keeps its timing, but not a failure."""
    job = _cancelled_after(TEST_JOB_TIMEOUT_MINUTES - 1)
    assert not timed_out(job)
    assert job_status(job) is Status.CANCELLED
    assert has_run(job)


def test_a_job_cancelled_while_queued_never_ran():
    """Without a runner the job never started, so however long it waited it is not a timeout."""
    job = make_workflow_job(
        conclusion=WorkflowJobConclusion.CANCELLED,
        runner_name=None,
        started_at="2026-01-01T10:00:00Z",
        completed_at="2026-01-01T14:00:00Z",
    )
    assert not timed_out(job)
    assert not has_run(job)
    assert job_status(job) is Status.CANCELLED


def test_a_job_concluded_timed_out_is_a_timeout_whatever_it_ran():
    """The API was never seen sending it, but a literal `timed_out` conclusion needs no runner."""
    job = make_workflow_job(conclusion=WorkflowJobConclusion.TIMED_OUT, runner_name=None)
    assert timed_out(job)
    assert job_status(job) is Status.FAILURE


@pytest.mark.parametrize(
    ("conclusion", "timeout_stop", "expected"),
    [
        pytest.param("cancelled", False, Status.CANCELLED, id="genuine-cancellation"),
        pytest.param("cancelled", True, Status.FAILURE, id="timeout-cancellation"),
        pytest.param("failure", True, Status.FAILURE, id="failed-run"),
        pytest.param("success", False, Status.SUCCESS, id="successful-run"),
    ],
)
def test_batch_status(conclusion: str, timeout_stop: bool, expected: Status):
    """Only a detected timeout reclassifies a cancelled run; every other conclusion stands."""
    jobs = [_cancelled_after(TEST_JOB_TIMEOUT_MINUTES)] if timeout_stop else []
    assert batch_status(conclusion, jobs) is expected
