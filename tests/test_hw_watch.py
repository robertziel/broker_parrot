"""Hardware flight-recorder (migration 0021, ``queue_workflows/hw_watch.py``).

Two-tier persisted hardware telemetry — the productized descendant of the
ad-hoc "box-b blackbox" scripts that caught a GB10 box's firmware thermal
kill (HW slowdown 0x48 asserted 7 s before an OS-invisible power cut):

* ``detail``  tier — super-detailed samples (default every 2 s), retained 1 h.
* ``history`` tier — coarse samples (default every 60 s), retained 24 h.

The write path is best-effort telemetry (own connection, swallow-on-failure)
like ``record_node_event``: a sample blip must never take down a worker. The
cadence brain (``HwWatchRecorder``) is pure logic with injectable ``now_fn`` /
``record_fn`` / ``sample_fn`` seams so it's tested on a virtual clock with no
real waiting, matching the repo's seam convention.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from queue_workflows import hw_watch


def _utc(offset_s: float = 0.0) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=offset_s)


# ── storage: record / read / prune ───────────────────────────────────────


def test_record_and_recent_roundtrip():
    payload = {"gpu": {"temp_c": 84, "throttle_hex": "0x48"}, "load1": 1.31}
    rid = hw_watch.record_hw_sample(
        host_label="boxa-gpu", tier="detail", data=payload, box="boxa",
    )
    assert rid is not None
    hw_watch.record_hw_sample(host_label="boxa-gpu", tier="history", data={"h": 1})

    rows = hw_watch.recent_hw_samples(tier="detail")
    assert len(rows) == 1
    row = rows[0]
    assert row["host_label"] == "boxa-gpu"
    assert row["box"] == "boxa"
    assert row["tier"] == "detail"
    # data round-trips as a real dict on both backends (sqlite stores TEXT).
    assert row["data"]["gpu"]["throttle_hex"] == "0x48"
    assert row["data"]["load1"] == 1.31

    assert len(hw_watch.recent_hw_samples(tier="history")) == 1


def test_recent_filters_by_host_and_window():
    hw_watch.record_hw_sample(host_label="a", tier="detail", data={})
    hw_watch.record_hw_sample(host_label="b", tier="detail", data={})
    hw_watch.record_hw_sample(
        host_label="a", tier="detail", data={}, created_at=_utc(-2 * 3600),
    )

    assert len(hw_watch.recent_hw_samples(host_label="a", tier="detail")) == 1
    assert len(hw_watch.recent_hw_samples(tier="detail")) == 2
    assert len(
        hw_watch.recent_hw_samples(host_label="a", tier="detail", since_s=3 * 3600)
    ) == 2


def test_prune_two_tier_retention():
    # detail: fresh survives, 2 h old dies (1 h retention).
    hw_watch.record_hw_sample(host_label="x", tier="detail", data={})
    hw_watch.record_hw_sample(
        host_label="x", tier="detail", data={}, created_at=_utc(-2 * 3600),
    )
    # history: 2 h old survives, 25 h old dies (24 h retention).
    hw_watch.record_hw_sample(
        host_label="x", tier="history", data={}, created_at=_utc(-2 * 3600),
    )
    hw_watch.record_hw_sample(
        host_label="x", tier="history", data={}, created_at=_utc(-25 * 3600),
    )

    detail_deleted, history_deleted = hw_watch.prune_hw_watch()
    assert (detail_deleted, history_deleted) == (1, 1)
    assert len(hw_watch.recent_hw_samples(tier="detail", since_s=48 * 3600)) == 1
    assert len(hw_watch.recent_hw_samples(tier="history", since_s=48 * 3600)) == 1


def test_record_is_best_effort_never_raises(monkeypatch):
    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(hw_watch, "connection", _boom)
    assert (
        hw_watch.record_hw_sample(host_label="x", tier="detail", data={"a": 1})
        is None
    )


def test_prune_swallows_storage_errors(monkeypatch):
    def _boom():
        raise RuntimeError("no such table: hw_watch_samples")

    monkeypatch.setattr(hw_watch, "connection", _boom)
    assert hw_watch.prune_hw_watch() == (0, 0)


# ── cadence brain (virtual clock, no waiting) ────────────────────────────


def test_recorder_cadence_on_a_virtual_clock():
    wrote: list[tuple[str, float]] = []
    sampled: list[float] = []
    clock = {"t": 1000.0}

    def _sample():
        sampled.append(clock["t"])
        return {"n": len(sampled)}

    def _record(**kw):
        wrote.append((kw["tier"], clock["t"]))
        return 1

    rec = hw_watch.HwWatchRecorder(
        host_label="boxa-gpu",
        detail_interval_s=2.0,
        history_interval_s=60.0,
        record_fn=_record,
        sample_fn=_sample,
    )

    def tick_at(t: float) -> list[str]:
        clock["t"] = t
        return rec.tick(now=t)

    assert sorted(tick_at(1000.0)) == ["detail", "history"]  # first tick: both
    assert tick_at(1001.0) == []                             # nothing due
    assert tick_at(1002.0) == ["detail"]                     # detail cadence
    assert tick_at(1003.9) == []
    assert tick_at(1004.1) == ["detail"]
    assert tick_at(1060.0) == ["detail", "history"]          # history cadence
    # ONE deep-sample per tick that wrote anything — shared across tiers.
    assert len(sampled) == 4


def test_recorder_records_nothing_when_no_tier_due():
    calls = []
    rec = hw_watch.HwWatchRecorder(
        host_label="h",
        detail_interval_s=10.0,
        history_interval_s=10.0,
        record_fn=lambda **kw: calls.append(kw) or 1,
        sample_fn=lambda: {"s": 1},
    )
    rec.tick(now=0.0)
    assert len(calls) == 2
    rec.tick(now=1.0)
    assert len(calls) == 2  # sample_fn must not even run — checked via record


# ── deep sample (never raises; stable shape) ─────────────────────────────


def test_deep_sample_has_stable_shape_even_when_probes_fail(monkeypatch):
    for probe in (
        "_gpu_deep", "_read_thermal_zones", "_read_hwmon_temps",
        "_read_meminfo", "_read_load1", "_disk_root_used_mb", "_cpu_stats",
    ):
        monkeypatch.setattr(
            hw_watch, probe,
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("probe down")),
        )
    sample = hw_watch.deep_sample()
    for key in ("gpu", "tz", "hwmon", "mem", "load1", "disk_root_used_mb", "cpu"):
        assert key in sample


# ── CPU hardware stats ───────────────────────────────────────────────────
#
# On a unified-memory SoC (GB10) the CPU shares the die + power budget with
# the GPU, so CPU temp / freq / throttle are part of the same box-health
# story. All stdlib (/proc, /sys) — no psutil dependency for the recorder.


def test_cpu_percent_from_jiffy_delta():
    # idle unchanged (+0), total +100 → 100% busy
    seq = iter([(1000, 2000), (1000, 2100)])
    assert hw_watch._read_cpu_percent(interval=0.0, snapshot=lambda: next(seq)) == 100.0
    # idle +100, total +200 → 50% busy
    seq = iter([(1000, 2000), (1100, 2200)])
    assert hw_watch._read_cpu_percent(interval=0.0, snapshot=lambda: next(seq)) == 50.0


def test_cpu_percent_none_when_no_time_passed():
    seq = iter([(1000, 2000), (1000, 2000)])   # dt = 0
    assert hw_watch._read_cpu_percent(interval=0.0, snapshot=lambda: next(seq)) is None


def test_cpu_stats_has_stable_shape():
    stats = hw_watch._cpu_stats()
    for key in ("percent", "freq_mhz", "temp_c", "throttle_count", "ncpu"):
        assert key in stats
    assert stats["ncpu"] is None or stats["ncpu"] >= 1


def test_cpu_stats_survives_sub_probe_failure(monkeypatch):
    for p in ("_read_cpu_percent", "_read_cpu_freq_mhz", "_read_cpu_temp_c",
              "_read_cpu_throttle_count"):
        monkeypatch.setattr(hw_watch, p,
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    stats = hw_watch._cpu_stats()          # must not raise
    assert stats["percent"] is None and stats["temp_c"] is None


def test_deep_sample_includes_cpu(monkeypatch):
    monkeypatch.setattr(hw_watch, "_cpu_stats",
                        lambda: {"percent": 42.0, "temp_c": 61.0, "freq_mhz": 3200,
                                 "throttle_count": 0, "ncpu": 20})
    sample = hw_watch.deep_sample()
    assert sample["cpu"]["percent"] == 42.0 and sample["cpu"]["temp_c"] == 61.0


def test_deep_sample_collects_real_probe_data():
    sample = hw_watch.deep_sample()
    assert isinstance(sample, dict)
    assert isinstance(sample.get("tz"), list)
    assert isinstance(sample.get("mem"), dict)


# ── throttle index (mask → 0-100 severity + decoded reasons) ─────────────
#
# The raw NVML throttle-reason mask (``0x0000000000000048``) is the smoking-gun
# field but doesn't graph. ``throttle_index`` collapses it to a 0-100 severity
# (max over active bits — one HW-thermal bit already means "at the wall"), and
# ``throttle_reasons`` decodes the bit names for tooltips. Calibration from the
# GB10 incident: the box died seconds after 0x48 (HW slowdown + HW brake).


def test_throttle_index_severity_ladder():
    assert hw_watch.throttle_index("0x0000000000000000") == 0
    assert hw_watch.throttle_index("0x0000000000000001") == 0    # GPU idle: benign
    assert hw_watch.throttle_index("0x0000000000000004") == 30   # SW power cap
    assert hw_watch.throttle_index("0x0000000000000020") == 50   # SW thermal
    assert hw_watch.throttle_index("0x0000000000000008") == 85   # HW slowdown
    assert hw_watch.throttle_index("0x0000000000000080") == 90   # HW power brake
    assert hw_watch.throttle_index("0x0000000000000040") == 100  # HW thermal
    # Combined mask scores its WORST bit (the incident's 0x48 = HW slowdown
    # + SW power cap... 0x40|0x08): max wins.
    assert hw_watch.throttle_index("0x0000000000000048") == 100
    assert hw_watch.throttle_index("0x0000000000000024") == 50


def test_throttle_index_handles_unknown_input():
    assert hw_watch.throttle_index(None) is None
    assert hw_watch.throttle_index("") is None
    assert hw_watch.throttle_index("[N/A]") is None
    # Unknown high bits alone still score 0 (nothing recognised active).
    assert hw_watch.throttle_index("0x0000000000000200") == 0


def test_throttle_reasons_decode_names():
    got = set(hw_watch.throttle_reasons("0x0000000000000048"))
    assert got == {"hw_slowdown", "hw_thermal"}
    assert hw_watch.throttle_reasons("0x0000000000000020") == ["sw_thermal"]
    assert hw_watch.throttle_reasons("0x0000000000000000") == []
    assert hw_watch.throttle_reasons(None) == []


def test_deep_sample_carries_throttle_index(monkeypatch):
    monkeypatch.setattr(hw_watch, "_gpu_deep", lambda: [
        {"temp_c": 84, "throttle_hex": "0x0000000000000048"},
        {"temp_c": 70, "throttle_hex": "0x0000000000000000"},
    ])
    sample = hw_watch.deep_sample()
    assert sample["gpu"][0]["throttle_index"] == 100
    assert sample["gpu"][1]["throttle_index"] == 0
    assert sample["throttle_index"] == 100  # top-level = worst GPU (graph feed)


# ── power-brake detection (the PSU watchdog signal) ──────────────────────
#
# GB10 exposes NO input-rail telemetry (no power_supply/typec/regulator
# surfaces — verified on the incident box), so the GPU-side brake bits are
# the only software-visible echo of a power-delivery event: 0x08 hw_slowdown
# (NVML: thermal OR *external power brake assertion*) and 0x80 hw_power_brake.
# "Pure" = brake with no thermal bit alongside — the power-side signature.


def test_power_brake_bits():
    assert hw_watch.power_brake("0x0000000000000008") is True
    assert hw_watch.power_brake("0x0000000000000080") is True
    assert hw_watch.power_brake("0x0000000000000048") is True   # thermal+brake
    assert hw_watch.power_brake("0x0000000000000020") is False
    assert hw_watch.power_brake("0x0000000000000000") is False
    assert hw_watch.power_brake(None) is None


def test_power_brake_pure_excludes_thermal_company():
    assert hw_watch.power_brake_pure("0x0000000000000008") is True
    assert hw_watch.power_brake_pure("0x0000000000000088") is True
    assert hw_watch.power_brake_pure("0x0000000000000048") is False  # 0x40 thermal
    assert hw_watch.power_brake_pure("0x0000000000000028") is False  # 0x20 thermal
    assert hw_watch.power_brake_pure("0x0000000000000000") is False
    assert hw_watch.power_brake_pure(None) is None


_PERF_BLOCK = b"""
GPU 0000000F:01:00.0
    Clocks Event Reasons
        SW Power Cap                                   : Not Active
        SW Thermal Slowdown                            : Active
    Violations
        SW Power Capping                               : 81754120 us
        SW Thermal Slowdown                            : 620477037 us
        HW Thermal Slowdown                            : 149082138 us
        HW Power Braking                               : 0 us
"""


def test_gpu_violations_parses_accumulated_counters(monkeypatch):
    monkeypatch.setattr(hw_watch.subprocess, "check_output", lambda *a, **k: _PERF_BLOCK)
    vio = hw_watch._gpu_violations()
    assert vio == {
        "sw_power_us": 81754120,
        "sw_thermal_us": 620477037,
        "hw_thermal_us": 149082138,
        "hw_power_brake_us": 0,
    }
    # the "Not Active"/"Active" clock-event lines must NOT be misread as counters
    assert "sw_power_cap" not in vio


def test_deep_sample_attaches_violations(monkeypatch):
    monkeypatch.setattr(hw_watch, "_gpu_deep", lambda: [{"temp_c": 84, "throttle_hex": "0x0"}])
    monkeypatch.setattr(hw_watch, "_gpu_violations", lambda: {"hw_power_brake_us": 250000})
    sample = hw_watch.deep_sample()
    assert sample["gpu"][0]["violations"]["hw_power_brake_us"] == 250000


def test_deep_sample_survives_violation_probe_failure(monkeypatch):
    monkeypatch.setattr(hw_watch, "_gpu_deep", lambda: [{"temp_c": 84, "throttle_hex": "0x0"}])
    monkeypatch.setattr(hw_watch, "_gpu_violations",
                        lambda: (_ for _ in ()).throw(RuntimeError("nvidia-smi gone")))
    sample = hw_watch.deep_sample()          # must not raise
    assert "violations" not in sample["gpu"][0]


def test_nvidia_deep_parses_instantaneous_power(monkeypatch):
    csv = b"84, 73.10, 112.78, 2405, 96, 0, P0, 0x0000000000000048, [N/A], [N/A]\n"
    monkeypatch.setattr(hw_watch.subprocess, "check_output", lambda *a, **k: csv)
    gpus = hw_watch._nvidia_deep()
    assert len(gpus) == 1
    g = gpus[0]
    assert g["power_w"] == 73.1 and g["power_inst_w"] == 112.78
    assert g["throttle_hex"] == "0x0000000000000048"
    assert g["vram_used_mb"] is None                 # [N/A] on unified parts


# ── env knobs (envcompat: canonical + legacy twin) ───────────────────────


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("QUEUE_WORKFLOWS_HW_WATCH", raising=False)
    monkeypatch.delenv("AI_LEADS_HW_WATCH", raising=False)
    assert hw_watch.enabled() is False


def test_enabled_via_canonical_and_legacy_env(monkeypatch):
    monkeypatch.delenv("AI_LEADS_HW_WATCH", raising=False)
    monkeypatch.setenv("QUEUE_WORKFLOWS_HW_WATCH", "1")
    assert hw_watch.enabled() is True

    monkeypatch.delenv("QUEUE_WORKFLOWS_HW_WATCH", raising=False)
    monkeypatch.setenv("AI_LEADS_HW_WATCH", "1")
    assert hw_watch.enabled() is True


def test_interval_and_retention_knobs(monkeypatch):
    monkeypatch.setenv("QUEUE_WORKFLOWS_HW_WATCH_DETAIL_INTERVAL_S", "0.5")
    monkeypatch.setenv("QUEUE_WORKFLOWS_HW_WATCH_HISTORY_INTERVAL_S", "30")
    monkeypatch.setenv("QUEUE_WORKFLOWS_HW_WATCH_DETAIL_RETENTION_S", "1800")
    monkeypatch.setenv("QUEUE_WORKFLOWS_HW_WATCH_HISTORY_RETENTION_S", "43200")
    assert hw_watch.detail_interval_s() == 0.5
    assert hw_watch.history_interval_s() == 30.0
    assert hw_watch.detail_retention_s() == 1800
    assert hw_watch.history_retention_s() == 43200


def test_interval_knobs_have_sane_defaults(monkeypatch):
    for var in (
        "QUEUE_WORKFLOWS_HW_WATCH_DETAIL_INTERVAL_S",
        "QUEUE_WORKFLOWS_HW_WATCH_HISTORY_INTERVAL_S",
        "QUEUE_WORKFLOWS_HW_WATCH_DETAIL_RETENTION_S",
        "QUEUE_WORKFLOWS_HW_WATCH_HISTORY_RETENTION_S",
    ):
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(var.replace("QUEUE_WORKFLOWS", "AI_LEADS"), raising=False)
    assert hw_watch.detail_interval_s() == 2.0
    assert hw_watch.history_interval_s() == 60.0
    assert hw_watch.detail_retention_s() == 3600
    assert hw_watch.history_retention_s() == 86400
