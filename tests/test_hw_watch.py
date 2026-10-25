"""Hardware flight-recorder (migration 0021, ``queue_workflows/hw_watch.py``).

Two-tier persisted hardware telemetry — built to catch a
firmware thermal kill on a unified-memory GPU box (an HW slowdown bit asserted seconds
before an OS-invisible power cut):

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

from queue_workflows import envcompat, hw_watch


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
        "_net_stats",
    ):
        monkeypatch.setattr(
            hw_watch, probe,
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("probe down")),
        )
    sample = hw_watch.deep_sample()
    for key in ("gpu", "tz", "hwmon", "mem", "load1", "disk_root_used_mb", "cpu",
                "net", "net_rx_bytes", "net_tx_bytes", "net_speed_mbps"):
        assert key in sample


# ── CPU hardware stats ───────────────────────────────────────────────────
#
# On a unified-memory SoC (unified-memory GPU box) the CPU shares the die + power budget with
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
# ``throttle_reasons`` decodes the bit names for tooltips. Calibration: a unified-memory GPU
# box can die seconds after 0x48 (HW slowdown + HW thermal).


def test_throttle_index_severity_ladder():
    assert hw_watch.throttle_index("0x0000000000000000") == 0
    assert hw_watch.throttle_index("0x0000000000000001") == 0    # GPU idle: benign
    assert hw_watch.throttle_index("0x0000000000000004") == 30   # SW power cap
    assert hw_watch.throttle_index("0x0000000000000020") == 50   # SW thermal
    assert hw_watch.throttle_index("0x0000000000000008") == 85   # HW slowdown
    assert hw_watch.throttle_index("0x0000000000000080") == 90   # HW power brake
    assert hw_watch.throttle_index("0x0000000000000040") == 100  # HW thermal
    # Combined mask scores its WORST bit (0x48 = HW thermal 0x40
    # | HW slowdown 0x08): max wins.
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
# a unified-memory GPU box may expose NO input-rail telemetry (no power_supply/typec/regulator
# surfaces), so the GPU-side brake bits are
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


# ── network hardware stats (/proc/net/dev + /sys/class/net) ──────────────
#
# Network throughput is a box-health signal like CPU/GPU load: a stalled model
# pull, a box that dropped off the network, or a NIC that renegotiated to a
# lower link speed all show here. /proc/net/dev's byte columns are MONOTONIC
# counters (like the GPU violation µs), so throughput is the delta between two
# samples — stored raw, derived at read time via ``net_rates`` — which keeps
# the sample stateless and lets a reader reconstruct the rate at any window.

_PROC_NET_DEV = """\
Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo: 1000000    5000    0    0    0     0          0         0  1000000    5000    0    0    0     0       0          0
  eth0: 900000000 700000    1    2    0     0          0        10 120000000  400000    0    3    0     0       0          0
 wlan0:       0        0    0    0    0     0          0         0        0        0    0    0    0     0       0          0
garbage row without the expected column count
"""


def test_parse_net_dev_extracts_counters_and_drops_loopback():
    devs = hw_watch._parse_net_dev(_PROC_NET_DEV)
    assert "lo" not in devs                       # loopback = intra-host noise
    assert set(devs) == {"eth0", "wlan0"}
    e = devs["eth0"]
    assert e["rx_bytes"] == 900000000 and e["tx_bytes"] == 120000000
    assert e["rx_packets"] == 700000 and e["tx_packets"] == 400000
    assert e["rx_errs"] == 1 and e["tx_drop"] == 3


def test_parse_net_dev_tolerates_garbage():
    assert hw_watch._parse_net_dev("") == {}
    assert hw_watch._parse_net_dev("h1\nh2\nfoo: 1 2 3\n") == {}   # <16 columns


def test_net_rates_derive_throughput_from_monotonic_counters():
    prev = {"net_rx_bytes": 1000, "net_tx_bytes": 500}
    curr = {"net_rx_bytes": 3000, "net_tx_bytes": 1500}
    r = hw_watch.net_rates(prev, curr, dt_s=2.0)
    assert r["rx_bytes_per_s"] == 1000.0          # (3000-1000)/2
    assert r["tx_bytes_per_s"] == 500.0           # (1500-500)/2


def test_net_rates_gap_reset_or_missing_reads_none_not_a_zero():
    # counter went BACKWARDS (reboot / iface reset) → None, never a negative rate
    assert hw_watch.net_rates(
        {"net_rx_bytes": 5000}, {"net_rx_bytes": 10}, dt_s=2.0
    )["rx_bytes_per_s"] is None
    # non-positive dt → None
    assert hw_watch.net_rates(
        {"net_rx_bytes": 0}, {"net_rx_bytes": 10}, dt_s=0
    )["rx_bytes_per_s"] is None
    # missing counter / missing sample → None
    assert hw_watch.net_rates(
        {}, {"net_rx_bytes": 10}, dt_s=2.0
    )["rx_bytes_per_s"] is None
    assert hw_watch.net_rates(
        None, {"net_rx_bytes": 10}, dt_s=2.0
    )["rx_bytes_per_s"] is None


def test_net_iface_meta_reads_link_speed_and_carrier(tmp_path):
    base = tmp_path / "net"; base.mkdir()
    eth = base / "eth0"; eth.mkdir()
    (eth / "operstate").write_text("up\n")
    (eth / "carrier").write_text("1\n")
    (eth / "speed").write_text("1000\n")
    (eth / "device").mkdir()                      # a real NIC has device/
    assert hw_watch._net_iface_meta("eth0", base=str(base)) == {
        "operstate": "up", "carrier": True, "speed_mbps": 1000, "physical": True,
    }


def test_net_iface_meta_virtual_iface_reports_no_real_speed(tmp_path):
    base = tmp_path / "net"; base.mkdir()
    d = base / "docker0"; d.mkdir()
    (d / "operstate").write_text("down\n")
    (d / "speed").write_text("-1\n")              # virtual/down → -1 → None
    meta = hw_watch._net_iface_meta("docker0", base=str(base))
    assert meta["speed_mbps"] is None and meta["physical"] is False
    assert meta["carrier"] is None                # no carrier file → None


def test_iface_virtual_classification():
    # loopback, container/bridge plumbing, and VPN/tunnels (double-count the wire)
    for v in ("lo", "docker0", "veth1a2b3c", "br-abc123", "virbr0", "cni0",
              "tailscale0", "wg0", "zt0"):
        assert hw_watch._iface_is_virtual(v), v
    for real in ("eth0", "enp3s0", "wlan0", "bond0"):   # bond = REAL aggregate
        assert not hw_watch._iface_is_virtual(real), real


def test_iface_is_enslaved_detects_bond_slave(tmp_path):
    base = tmp_path / "net"; base.mkdir()
    slave = base / "eth0"; slave.mkdir()
    (slave / "master").mkdir()                      # a bond slave has master/
    plain = base / "eth1"; plain.mkdir()
    assert hw_watch._iface_is_enslaved("eth0", base=str(base)) is True
    assert hw_watch._iface_is_enslaved("eth1", base=str(base)) is False


def _iface_counters(rx, tx):
    return {"rx_bytes": rx, "tx_bytes": tx, "rx_packets": 0, "tx_packets": 0,
            "rx_errs": 0, "tx_errs": 0, "rx_drop": 0, "tx_drop": 0}


def test_net_stats_lists_physical_and_headline_excludes_down_and_slaves(monkeypatch):
    monkeypatch.setattr(hw_watch, "_read_net_dev", lambda: {
        "bond0": _iface_counters(100, 40),        # aggregate → counted
        "eth0": _iface_counters(60, 25),          # bond slave → LISTED, not summed
        "eth1": _iface_counters(40, 15),          # bond slave → LISTED, not summed
        "wlan0": _iface_counters(7, 3),           # standalone up → counted
        "docker0": _iface_counters(9999, 9999),   # virtual → dropped entirely
        "eth2": _iface_counters(5, 5),            # down → LISTED, not summed
    })
    meta = {
        "bond0": {"operstate": "up", "carrier": True, "speed_mbps": 2000, "physical": False},
        "eth0": {"operstate": "up", "carrier": True, "speed_mbps": 1000, "physical": True},
        "eth1": {"operstate": "up", "carrier": True, "speed_mbps": 1000, "physical": True},
        "wlan0": {"operstate": "up", "carrier": True, "speed_mbps": 300, "physical": True},
        "eth2": {"operstate": "down", "carrier": False, "speed_mbps": None, "physical": True},
    }
    monkeypatch.setattr(hw_watch, "_net_iface_meta", lambda name, **k: meta[name])
    monkeypatch.setattr(hw_watch, "_iface_is_enslaved",
                        lambda name, **k: name in ("eth0", "eth1"))
    stats = hw_watch._net_stats()
    # headline: bond0 + wlan0 only (slaves already in bond0; eth2 down; docker0 virtual)
    assert stats["rx_bytes"] == 107 and stats["tx_bytes"] == 43
    assert stats["speed_mbps"] == 2000            # fastest non-enslaved up link
    # per-NIC view lists every PHYSICAL iface + bond (slaves + down included for
    # visibility); only the container plumbing (docker0) is dropped.
    listed = {i["iface"] for i in stats["ifaces"]}
    assert listed == {"bond0", "eth0", "eth1", "wlan0", "eth2"}
    assert "docker0" not in listed
    assert next(i for i in stats["ifaces"] if i["iface"] == "eth0")["enslaved"] is True


def test_deep_sample_includes_net_and_graph_fields(monkeypatch):
    monkeypatch.setattr(hw_watch, "_net_stats", lambda: {
        "ifaces": [{"iface": "eth0"}],
        "rx_bytes": 12345, "tx_bytes": 678, "speed_mbps": 1000,
    })
    sample = hw_watch.deep_sample()
    assert sample["net"]["rx_bytes"] == 12345
    # box-level, graph-ready (panel plots bytes/s = Δ over the row gap)
    assert sample["net_rx_bytes"] == 12345
    assert sample["net_tx_bytes"] == 678
    assert sample["net_speed_mbps"] == 1000


def test_deep_sample_collects_real_net_probe():
    sample = hw_watch.deep_sample()
    assert isinstance(sample.get("net"), dict)
    assert "net_rx_bytes" in sample and "net_speed_mbps" in sample


# ── env knobs (envcompat: canonical names) ───────────────────────────────


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("QUEUE_WORKFLOWS_HW_WATCH", raising=False)
    monkeypatch.delenv("QUEUE_WORKFLOWS_HW_WATCH", raising=False)
    assert hw_watch.enabled() is False


def test_enabled_via_canonical_env(monkeypatch):
    monkeypatch.delenv("QUEUE_WORKFLOWS_HW_WATCH", raising=False)
    monkeypatch.setenv("QUEUE_WORKFLOWS_HW_WATCH", "1")
    assert hw_watch.enabled() is True

    monkeypatch.delenv("QUEUE_WORKFLOWS_HW_WATCH", raising=False)
    monkeypatch.setenv("QUEUE_WORKFLOWS_HW_WATCH", "1")
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
        monkeypatch.delenv(envcompat.LEGACY_PREFIX + var[len(envcompat.CANONICAL_PREFIX):], raising=False)
    assert hw_watch.detail_interval_s() == 2.0
    assert hw_watch.history_interval_s() == 60.0
    assert hw_watch.detail_retention_s() == 3600
    assert hw_watch.history_retention_s() == 86400


# ── CPU/SoC temp: unified-memory GPU box (ARM) acpitz fallback ────────────────────────────────
#
# Regression fence for the blind spot that masked a CPU-thermal hard-power-off:
# _read_cpu_temp_c() only knew x86/AMD CPU sensors, so on the ARM unified-memory GPU box (whose SoC
# temp is exposed only via generic acpitz zones) it returned None → "CPU TEMP: no
# data" while the SoC was actually running hot.

def _fake_sysfs(monkeypatch, tmp_path, *, hwmon=None, zones=None):
    """Redirect /sys/class/{hwmon,thermal} globs to a tmp tree; real glob otherwise."""
    import glob as _g
    real = _g.glob
    hroot = tmp_path / "hwmon"; troot = tmp_path / "thermal"
    hroot.mkdir(); troot.mkdir()
    for i, (name, temps) in enumerate(hwmon or []):
        d = hroot / f"hwmon{i}"; d.mkdir()
        (d / "name").write_text(name)
        for j, t in enumerate(temps, 1):
            (d / f"temp{j}_input").write_text(str(t))
    for i, (ztype, temp) in enumerate(zones or []):
        d = troot / f"thermal_zone{i}"; d.mkdir()
        (d / "type").write_text(ztype); (d / "temp").write_text(str(temp))

    def fake(pat):
        p = (pat.replace("/sys/class/hwmon", str(hroot))
                .replace("/sys/class/thermal", str(troot)))
        return real(p)
    monkeypatch.setattr(hw_watch.glob, "glob", fake)


def test_cpu_temp_arm_soc_falls_back_to_hottest_acpitz(monkeypatch, tmp_path):
    # ARM unified-memory GPU box: no CPU hwmon, only generic acpitz zones.
    _fake_sysfs(monkeypatch, tmp_path,
                hwmon=[("acpi_fan", [])],
                zones=[("acpitz", 72000), ("acpitz", 95000), ("acpitz", 88000)])
    assert hw_watch._read_cpu_temp_c() == 95.0     # SoC package temp, no longer None


def test_cpu_temp_prefers_coretemp_over_acpitz_on_x86(monkeypatch, tmp_path):
    # x86: a real coretemp must win; the hot acpitz zone must NOT shadow it.
    _fake_sysfs(monkeypatch, tmp_path,
                hwmon=[("coretemp", [61000, 58000])],
                zones=[("acpitz", 99000)])
    assert hw_watch._read_cpu_temp_c() == 61.0


def test_cpu_temp_none_when_nothing_exposed(monkeypatch, tmp_path):
    _fake_sysfs(monkeypatch, tmp_path, hwmon=[], zones=[])
    assert hw_watch._read_cpu_temp_c() is None
