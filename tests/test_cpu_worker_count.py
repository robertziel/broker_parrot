"""``node_pool.cpu_worker_count`` defaults to the box's AVAILABLE cores.

The default per-box cpu-worker count is the box's cgroup-aware available CPU
cores (so each box scales to its own capacity), overridable via
``AI_LEADS_WORKFLOW_CPU_WORKERS``. ``_available_cpus`` is cgroup-aware because
workers run in containers — a CFS-quota- or cpuset-limited container must count
its real share, not the host's cores. Pure logic with injected seams (no DB).
"""

from __future__ import annotations

import pytest

from queue_workflows import node_pool


def _reader(files: dict[str, str]):
    """Fake ``read_text`` over a path→contents map; missing path → FileNotFoundError
    (mirrors a cgroup file that isn't present)."""
    def _read(path: str) -> str:
        if path in files:
            return files[path]
        raise FileNotFoundError(path)
    return _read


# ── _available_cpus: the detection chain ────────────────────────────────────


def test_cgroup_v2_quota_wins():
    r = _reader({"/sys/fs/cgroup/cpu.max": "400000 100000"})  # 4 cpus
    assert node_pool._available_cpus(
        read_text=r, affinity_fn=lambda: 99, cpu_count_fn=lambda: 99
    ) == 4


def test_cgroup_v2_max_unlimited_falls_through():
    r = _reader({"/sys/fs/cgroup/cpu.max": "max 100000"})  # no quota
    assert node_pool._available_cpus(
        read_text=r, affinity_fn=lambda: 6, cpu_count_fn=lambda: 99
    ) == 6


def test_cgroup_v2_sub_one_cpu_quota_is_one_not_host():
    # A 0.5-cpu container is authoritative: 1 worker, NOT the host core count.
    r = _reader({"/sys/fs/cgroup/cpu.max": "50000 100000"})
    assert node_pool._available_cpus(
        read_text=r, affinity_fn=lambda: 32, cpu_count_fn=lambda: 32
    ) == 1


def test_cgroup_v2_fractional_quota_floors():
    r = _reader({"/sys/fs/cgroup/cpu.max": "250000 100000"})  # 2.5 cpus → floor 2
    assert node_pool._available_cpus(
        read_text=r, affinity_fn=lambda: 99, cpu_count_fn=lambda: 99
    ) == 2


def test_cgroup_v1_quota_when_no_v2():
    r = _reader({
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "300000",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000",
    })
    assert node_pool._available_cpus(
        read_text=r, affinity_fn=lambda: 99, cpu_count_fn=lambda: 99
    ) == 3


def test_cgroup_v1_unset_quota_minus_one_falls_through():
    r = _reader({
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "-1",   # unlimited
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000",
    })
    assert node_pool._available_cpus(
        read_text=r, affinity_fn=lambda: 10, cpu_count_fn=lambda: 99
    ) == 10


def test_falls_back_to_affinity_when_no_cgroup():
    assert node_pool._available_cpus(
        read_text=_reader({}), affinity_fn=lambda: 12, cpu_count_fn=lambda: 99
    ) == 12


def test_falls_back_to_cpu_count_when_no_affinity():
    def _aff():
        raise OSError("no affinity")
    assert node_pool._available_cpus(
        read_text=_reader({}), affinity_fn=_aff, cpu_count_fn=lambda: 8
    ) == 8


def test_floor_is_one_when_everything_unknown():
    def _aff():
        raise OSError
    assert node_pool._available_cpus(
        read_text=_reader({}), affinity_fn=_aff, cpu_count_fn=lambda: None
    ) == 1


def test_garbage_cgroup_contents_do_not_crash():
    r = _reader({"/sys/fs/cgroup/cpu.max": "not-a-number"})
    assert node_pool._available_cpus(
        read_text=r, affinity_fn=lambda: 4, cpu_count_fn=lambda: 99
    ) == 4  # falls through cleanly to affinity


# ── cpu_worker_count: default = available cores, env overrides ──────────────


def test_cpu_worker_count_defaults_to_available_cpus(monkeypatch):
    monkeypatch.delenv("AI_LEADS_WORKFLOW_CPU_WORKERS", raising=False)
    assert node_pool.cpu_worker_count() == node_pool._available_cpus()
    assert node_pool.cpu_worker_count() >= 1


def test_cpu_worker_count_env_override_wins(monkeypatch):
    monkeypatch.setenv("AI_LEADS_WORKFLOW_CPU_WORKERS", "7")
    assert node_pool.cpu_worker_count() == 7


def test_cpu_worker_count_is_no_longer_hardcoded_five(monkeypatch):
    """Regression for the change: the default must track the box, not the old
    magic 5 (unless the box genuinely has 5 cores)."""
    monkeypatch.delenv("AI_LEADS_WORKFLOW_CPU_WORKERS", raising=False)
    assert node_pool.cpu_worker_count() == node_pool._available_cpus()


def test_cpu_worker_count_garbage_env_falls_back_to_available(monkeypatch):
    """A non-integer override must fall back to available cores, not the old
    hardcoded 5 (regression guard on _int_env's bad-value path)."""
    monkeypatch.setenv("AI_LEADS_WORKFLOW_CPU_WORKERS", "not-a-number")
    assert node_pool.cpu_worker_count() == node_pool._available_cpus()


def test_gpu_worker_count_default_unchanged(monkeypatch):
    monkeypatch.delenv("AI_LEADS_WORKFLOW_GPU_WORKERS", raising=False)
    assert node_pool.gpu_worker_count() == 1
