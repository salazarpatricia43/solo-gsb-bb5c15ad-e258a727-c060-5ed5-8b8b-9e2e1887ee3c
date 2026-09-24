"""Maintenance-slot queries: GET /jobs/{id}/maintenance-slots.

For a QUEUED job, every candidate start ``t`` in the half-open query window
``[start, end)`` predicts the run ``[t, t + duration_min)``; the start is
feasible when the fixed current RUNNING baseline, the job's own GPUs and the
summed saved maintenance reservation never exceed cluster capacity anywhere
in that run. The tests cover the service and HTTP levels: window merging,
endpoint touching, windows straddling the query end, isolated feasible
instants, the fixed RUNNING baseline and the non-QUEUED/unknown/error cases.
"""
import pytest

from app import db, scheduler
from tests.conftest import at, iso  # noqa: F401


def submit(tenant, gpus, minutes, weight=1.0, when=None, depends_on=None):
    job = scheduler.submit_job(
        tenant, gpus, minutes, weight, depends_on=depends_on
    )
    if when is not None:
        db.get_conn().execute(
            "UPDATE jobs SET queued_at = ? WHERE id = ?", (when, job["id"])
        )
    return scheduler._get_job_or_404(db.get_conn(), job["id"])


def window(start_min, end_min, reserved):
    return scheduler.create_maintenance_window(
        at(start_min), at(end_min), reserved
    )


def slots(job_id, lo, hi):
    return scheduler.maintenance_slots(job_id, at(lo), at(hi))


def _plain(slot_list):
    return [
        (s["start_at"], s["start_inclusive"], s["end_at"], s["end_inclusive"])
        for s in slot_list
    ]


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------

def test_no_windows_whole_query_is_feasible(conn):
    job = submit("a", 2, 60, when=iso(0))
    r = slots(job["id"], 0, 200)
    assert r["job_id"] == "J000001"
    assert r["status"] == "QUEUED"
    assert r["gpus"] == 2 and r["duration_min"] == 60
    assert r["cluster_gpus"] == 8 and r["running_gpus"] == 0
    assert r["earliest_start_at"] == iso(0)
    assert _plain(r["slots"]) == [(iso(0), True, iso(200), False)]


def test_query_window_is_half_open_on_both_sides(conn):
    # Window [40,100) reserves all 8; a 60-min run starting anywhere before
    # 100 overlaps it (t in (-20,100) blocked). The only touching start is
    # t=100, which the query [0,100) excludes -> no slot.
    job = submit("a", 8, 60, when=iso(0))
    window(40, 100, 8)
    r = slots(job["id"], 0, 100)
    assert r["slots"] == []
    assert r["earliest_start_at"] is None


def test_job_touching_window_at_start_is_feasible(conn):
    # 8-GPU run [0,60) and window [60,90): t=0 merely touches -> feasible;
    # starts in (0,90) overlap; t>=90 is feasible again.
    job = submit("a", 8, 60, when=iso(0))
    window(60, 90, 8)
    r = slots(job["id"], 0, 200)
    assert _plain(r["slots"]) == [
        (iso(0), True, iso(0), True),       # isolated feasible instant {0}
        (iso(90), True, iso(200), False),
    ]
    assert r["earliest_start_at"] == iso(0)


def test_window_inside_horizon_cuts_feasible_region(conn):
    # Need all 8 GPUs for 60 min; window [30,90) reserves 2. Any positive
    # reservation blocks: starts in (30-60,90) = (-30,90) are infeasible.
    job = submit("a", 8, 60, when=iso(0))
    window(30, 90, 2)
    r = slots(job["id"], 0, 200)
    assert _plain(r["slots"]) == [(iso(90), True, iso(200), False)]
    assert r["earliest_start_at"] == iso(90)


def test_window_straddling_query_end_clips_slot_inside(conn):
    # Window [100,300) reserves 8; 60-min run: starts in (40,300) blocked.
    # Inside [0,200) only [0,40] survives, and 40 is included (touch).
    job = submit("a", 8, 60, when=iso(0))
    window(100, 300, 8)
    r = slots(job["id"], 0, 200)
    assert _plain(r["slots"]) == [(iso(0), True, iso(40), True)]
    assert r["earliest_start_at"] == iso(0)


def test_window_entirely_before_query_keeps_all_starts(conn):
    # Window [-100,-50) can only block starts in (-160,-50): nothing here.
    job = submit("a", 8, 60, when=iso(0))
    window(-100, -50, 8)
    r = slots(job["id"], 0, 200)
    assert _plain(r["slots"]) == [(iso(0), True, iso(200), False)]


def test_window_starting_before_query_can_block_inside(conn):
    # Window [-30,30): runs starting t in (-90,30) overlap; t=30 touches.
    job = submit("a", 8, 60, when=iso(0))
    window(-30, 30, 8)
    r = slots(job["id"], 0, 200)
    assert _plain(r["slots"]) == [(iso(30), True, iso(200), False)]
    assert r["earliest_start_at"] == iso(30)


# ---------------------------------------------------------------------------
# Window merging: overlapping add, touching do not
# ---------------------------------------------------------------------------

def test_touching_windows_do_not_add(conn):
    # Two 3-GPU windows meeting at t=30; a 3-GPU job (headroom 8) is blocked
    # only where summed reservation > 5. Touching windows never stack, so no
    # instant reaches 6 -> the whole query stays feasible.
    job = submit("a", 3, 60, when=iso(0))
    window(0, 30, 3)
    window(30, 60, 3)
    r = slots(job["id"], 0, 200)
    assert _plain(r["slots"]) == [(iso(0), True, iso(200), False)]


def test_overlapping_windows_add(conn):
    # Same windows genuinely overlapping: sum peaks at 6 > 5, blocking runs
    # that cross the overlap. Blocked stretch [20,40) forbids starts in
    # (20-60,40) = (-40,40).
    job = submit("a", 3, 60, when=iso(0))
    window(0, 40, 3)
    window(20, 60, 3)
    r = slots(job["id"], 0, 200)
    assert _plain(r["slots"]) == [(iso(40), True, iso(200), False)]


def test_blocked_segments_exactly_duration_apart_leave_isolated_point(conn):
    # Two blocking stretches [30,40) and [100,110) for a 1-GPU 60-min job
    # (8 GPUs reserved at each). Cuts are (-30,40) and (40,110): start 40
    # survives BOTH because its run [40,100) touches each window exactly.
    job = submit("a", 1, 60, when=iso(0))
    window(30, 40, 8)
    window(100, 110, 8)
    r = slots(job["id"], 0, 200)
    assert _plain(r["slots"]) == [
        (iso(40), True, iso(40), True),        # isolated feasible instant
        (iso(110), True, iso(200), False),
    ]
    assert r["earliest_start_at"] == iso(40)


def test_reservation_within_headroom_never_blocks(conn):
    # Running baseline 2 + job 2 = 4 leaves 4 GPUs for maintenance; a 3-GPU
    # reservation (even stacked 3+3=6... use a single window) fits: feasible.
    submit("r", 2, 120, when=iso(-10))
    scheduler.run_batch(now=at(-10))
    job = submit("a", 2, 60, when=iso(0))
    window(20, 80, 3)
    r = slots(job["id"], 0, 200)
    assert r["running_gpus"] == 2
    assert _plain(r["slots"]) == [(iso(0), True, iso(200), False)]


def test_stacked_reservation_against_baseline_blocks(conn):
    # Same baseline/job (headroom for maintenance = 4); stacked 3+3=7 > 4 on
    # the [20,40) overlap blocks starts in (-20,40).
    submit("r", 2, 120, when=iso(-10))
    scheduler.run_batch(now=at(-10))
    job = submit("a", 2, 60, when=iso(0))
    window(0, 40, 3)
    window(20, 80, 3)
    r = slots(job["id"], 0, 200)
    assert _plain(r["slots"]) == [(iso(40), True, iso(200), False)]


def test_zero_reservation_window_is_ignored(conn):
    job = submit("a", 8, 60, when=iso(0))
    window(0, 100, 0)
    r = slots(job["id"], 0, 200)
    assert _plain(r["slots"]) == [(iso(0), True, iso(200), False)]


def test_fixed_running_baseline_blocks_everything_when_cluster_full(conn):
    # 8 GPUs already RUNNING: no maintenance needed, no start is feasible.
    submit("r", 8, 120, when=iso(-10))
    scheduler.run_batch(now=at(-10))
    job = submit("a", 1, 60, when=iso(0))
    window(0, 200, 0)
    r = slots(job["id"], 0, 200)
    assert r["running_gpus"] == 8
    assert r["slots"] == []
    assert r["earliest_start_at"] is None


def test_baseline_is_fixed_running_completion_not_speculated(conn):
    # The RUNNING job's *actual* end is not modelled: after a window that
    # would fit if it released, slots stay empty (capacity-only view).
    runner = submit("r", 7, 60, when=iso(0))
    scheduler.run_batch(now=at(0))
    job = submit("a", 2, 60, when=iso(0))
    # No maintenance at all, yet only 1 GPU free -> no feasible start,
    # even beyond the runner's duration_min (60).
    r = slots(job["id"], 0, 500)
    assert r["slots"] == []
    # Completing the runner immediately opens the whole query window.
    scheduler.complete_job(runner["id"])
    r2 = slots(job["id"], 0, 500)
    assert _plain(r2["slots"]) == [(iso(0), True, iso(500), False)]
    assert r2["running_gpus"] == 0


def test_other_queued_jobs_are_not_counted(conn):
    # QUEUED jobs neither contribute to the baseline nor compete; only the
    # candidate and RUNNING GPUs count.
    submit("q", 8, 60, when=iso(0))          # stays QUEUED
    job = submit("a", 8, 60, when=iso(0))
    r = slots(job["id"], 0, 200)
    assert r["running_gpus"] == 0
    assert _plain(r["slots"]) == [(iso(0), True, iso(200), False)]


def test_dependency_blocked_queued_job_still_reports_capacity_slots(conn):
    # Capacity view only: an unmet prerequisite does not suppress slots.
    dep = submit("d", 1, 60, when=iso(0))
    job = submit("a", 8, 60, when=iso(0), depends_on=[dep["id"]])
    assert scheduler.get_job(job["id"])["blocked_by"] != []
    r = slots(job["id"], 0, 200)
    assert _plain(r["slots"]) == [(iso(0), True, iso(200), False)]


# ---------------------------------------------------------------------------
# Status / validation errors
# ---------------------------------------------------------------------------

def test_unknown_job_raises_404(conn):
    with pytest.raises(scheduler.NotFound):
        slots("J999999", 0, 200)


@pytest.mark.parametrize("transition", ["run", "complete", "cancel"])
def test_only_queued_jobs_report_slots(conn, transition):
    job = submit("a", 2, 60, when=iso(0))
    if transition == "run":
        scheduler.run_batch(now=at(0))
    elif transition == "complete":
        scheduler.run_batch(now=at(0))
        scheduler.complete_job(job["id"])
    else:
        scheduler.cancel_job(job["id"])
    with pytest.raises(scheduler.Conflict):
        slots(job["id"], 0, 200)


def test_inverted_query_window_raises_400(conn):
    job = submit("a", 2, 60, when=iso(0))
    with pytest.raises(scheduler.SchedulerError, match="end must be later"):
        slots(job["id"], 100, 100)
    with pytest.raises(scheduler.SchedulerError, match="end must be later"):
        slots(job["id"], 100, 50)


def test_naive_datetimes_treated_as_utc(conn):
    from datetime import datetime

    job = submit("a", 8, 60, when=iso(0))
    window(30, 90, 2)
    r = scheduler.maintenance_slots(
        job["id"],
        datetime(2026, 1, 1, 0, 0),
        datetime(2026, 1, 1, 3, 20),
    )
    assert _plain(r["slots"]) == [(iso(90), True, iso(200), False)]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def test_maintenance_slots_http(client):
    client.post("/jobs", json={
        "tenant_id": "a", "gpus": 8, "duration_min": 60})
    client.post("/maintenance-windows", json={
        "start_at": "2026-01-01T00:30:00Z",
        "end_at": "2026-01-01T01:30:00Z",
        "reserved_gpus": 2})

    r = client.get("/jobs/J000001/maintenance-slots", params={
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-01T03:20:00Z"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == "J000001"
    assert body["status"] == "QUEUED"
    assert body["gpus"] == 8 and body["duration_min"] == 60
    assert body["cluster_gpus"] == 8 and body["running_gpus"] == 0
    assert body["earliest_start_at"] == "2026-01-01T01:30:00+00:00"
    assert body["slots"] == [{
        "start_at": "2026-01-01T01:30:00+00:00",
        "start_inclusive": True,
        "end_at": "2026-01-01T03:20:00+00:00",
        "end_inclusive": False,
    }]


def test_maintenance_slots_isolated_point_http(client):
    client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 60})
    client.post("/maintenance-windows", json={
        "start_at": "2026-01-01T00:30:00Z",
        "end_at": "2026-01-01T00:40:00Z",
        "reserved_gpus": 8})
    client.post("/maintenance-windows", json={
        "start_at": "2026-01-01T01:40:00Z",
        "end_at": "2026-01-01T01:50:00Z",
        "reserved_gpus": 8})

    r = client.get("/jobs/J000001/maintenance-slots", params={
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-01T03:20:00Z"})
    assert r.status_code == 200, r.text
    assert r.json()["slots"] == [
        {
            "start_at": "2026-01-01T00:40:00+00:00",
            "start_inclusive": True,
            "end_at": "2026-01-01T00:40:00+00:00",
            "end_inclusive": True,
        },
        {
            "start_at": "2026-01-01T01:50:00+00:00",
            "start_inclusive": True,
            "end_at": "2026-01-01T03:20:00+00:00",
            "end_inclusive": False,
        },
    ]
    assert r.json()["earliest_start_at"] == "2026-01-01T00:40:00+00:00"


def test_maintenance_slots_http_naive_and_offset_normalisation(client):
    client.post("/jobs", json={
        "tenant_id": "a", "gpus": 8, "duration_min": 60})
    client.post("/maintenance-windows", json={
        "start_at": "2026-01-01T00:30:00Z",
        "end_at": "2026-01-01T01:30:00Z",
        "reserved_gpus": 2})

    # Naive timestamps are read as UTC; +02:00 offsets are normalised too.
    r = client.get("/jobs/J000001/maintenance-slots", params={
        "start": "2026-01-01T00:00:00",
        "end": "2026-01-01T03:20:00+02:00"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["query_start_at"] == "2026-01-01T00:00:00+00:00"
    assert body["query_end_at"] == "2026-01-01T01:20:00+00:00"
    assert body["slots"] == []          # window ends at 01:30 > query end


def test_maintenance_slots_http_errors(client):
    client.post("/jobs", json={
        "tenant_id": "a", "gpus": 8, "duration_min": 60})
    client.post("/maintenance-windows", json={
        "start_at": "2026-01-01T00:30:00Z",
        "end_at": "2026-01-01T01:30:00Z",
        "reserved_gpus": 2})

    # Unknown job -> 404.
    r = client.get("/jobs/J000042/maintenance-slots", params={
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-01T03:20:00Z"})
    assert r.status_code == 404

    # Inverted window -> 400.
    r = client.get("/jobs/J000001/maintenance-slots", params={
        "start": "2026-01-01T03:20:00Z",
        "end": "2026-01-01T00:00:00Z"})
    assert r.status_code == 400
    assert "end must be later" in r.json()["detail"]

    # Malformed / missing datetimes -> 422.
    r = client.get("/jobs/J000001/maintenance-slots", params={
        "start": "nope", "end": "2026-01-01T03:20:00Z"})
    assert r.status_code == 422
    r = client.get("/jobs/J000001/maintenance-slots", params={
        "start": "2026-01-01T00:00:00Z"})
    assert r.status_code == 422

    # Once the job has run, the endpoint refuses with 409.
    client.post("/scheduler/batch", json={"now": "2026-02-01T00:00:00Z"})
    r = client.get("/jobs/J000001/maintenance-slots", params={
        "start": "2026-02-01T00:00:00Z",
        "end": "2026-02-01T03:20:00Z"})
    assert r.status_code == 409
