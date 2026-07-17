"""Hardware flight recorder — persisted two-tier box telemetry (migration 0021).

WHY. :mod:`queue_workflows.hw_metrics` is push-only — every sample is a
``NOTIFY`` and gone; nothing is retained. When a GPU box hard-dies there is no
hardware trail to autopsy. The motivating incident: a GB10 box's firmware
thermal protection killed the machine on every ``sdxl`` render with ZERO
kernel trace — the GPU asserted a hardware slowdown (throttle mask ``0x48``)
seconds before an OS-invisible power cut, at a board-zone temperature the OS
exposes no trip points for. Only an ad-hoc sync-per-line shell recorder caught
it. This module is that recorder, engine-owned and DB-persisted:

* ``detail``  tier — super-detailed samples (default every **2 s**), retained
  **1 h**. The death-second forensics ring.
* ``history`` tier — coarse samples (default every **60 s**), retained
  **24 h**. The "was it trending hot all afternoon?" ring.

WHAT A SAMPLE HOLDS (``deep_sample``): per-GPU temp / power draw / SM clock /
utilisation / pstate / **throttle-reason mask** (the smoking-gun field —
``0x20`` SW-thermal, ``0x40`` HW-thermal, ``0x08`` HW-brake) via nvidia-smi or
rocm-smi, every ``/sys/class/thermal`` zone, every hwmon temp (NVMe / NIC /
SoC), ``/proc/meminfo`` (unified-memory boxes live and die by host RAM),
loadavg, and root-fs usage. Every probe is best-effort and vendor-guarded — a
box without a GPU CLI just records the zones it has.

DESIGN CONTRACTS (match the repo's existing shapes):

* Writes are **best-effort telemetry** on their own connection, swallowing
  every exception (the ``record_node_event`` pattern) — a sample blip can
  never take down a worker, and a pre-0021 DB simply records nothing.
* The cadence brain (:class:`HwWatchRecorder`) is pure logic with injectable
  ``now_fn`` / ``record_fn`` / ``sample_fn`` seams — unit-tested on a virtual
  clock with no real waiting.
* Retention pruning (:func:`prune_hw_watch`) is dialect-portable (pg +
  sqlite) and runs on a NodePool sweep; the standalone CLI prunes for itself.
* Env knobs read through :func:`queue_workflows.envcompat.env_get` with
  canonical ``QUEUE_WORKFLOWS_*`` names (legacy ``AI_LEADS_*`` twins free).
  **Persistence is OFF by default** (``QUEUE_WORKFLOWS_HW_WATCH=1`` enables)
  so existing deploys are byte-compatible.

Standalone use (a box under investigation, workers parked)::

    QUEUE_WORKFLOWS_DB_BACKEND=pg QUEUE_WORKFLOWS_DB_URL=postgresql://... \
        queue-hw-watch                     # records + prunes until killed
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime
from typing import Any, Callable

from queue_workflows.db import connection
from queue_workflows.dialect import get_dialect
from queue_workflows.envcompat import env_get

log = logging.getLogger(__name__)

TIER_DETAIL = "detail"
TIER_HISTORY = "history"

_ENABLE_ENV = "QUEUE_WORKFLOWS_HW_WATCH"
_DETAIL_INTERVAL_ENV = "QUEUE_WORKFLOWS_HW_WATCH_DETAIL_INTERVAL_S"
_HISTORY_INTERVAL_ENV = "QUEUE_WORKFLOWS_HW_WATCH_HISTORY_INTERVAL_S"
_DETAIL_RETENTION_ENV = "QUEUE_WORKFLOWS_HW_WATCH_DETAIL_RETENTION_S"
_HISTORY_RETENTION_ENV = "QUEUE_WORKFLOWS_HW_WATCH_HISTORY_RETENTION_S"

_DETAIL_INTERVAL_DEFAULT_S = 2.0
_HISTORY_INTERVAL_DEFAULT_S = 60.0
_DETAIL_RETENTION_DEFAULT_S = 3600
_HISTORY_RETENTION_DEFAULT_S = 86400


# ── env knobs ─────────────────────────────────────────────────────────────


def enabled() -> bool:
    """Persistence gate — OFF unless ``QUEUE_WORKFLOWS_HW_WATCH`` is truthy."""
    raw = (env_get(_ENABLE_ENV) or "").strip().lower()
    return raw not in ("", "0", "false", "no")


def _env_float(name: str, default: float) -> float:
    raw = (env_get(name) or "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
        return v if v > 0 else default
    except (TypeError, ValueError):
        log.warning("[hw-watch] %s=%r is not a number; using %s", name, raw, default)
        return default


def detail_interval_s() -> float:
    return _env_float(_DETAIL_INTERVAL_ENV, _DETAIL_INTERVAL_DEFAULT_S)


def history_interval_s() -> float:
    return _env_float(_HISTORY_INTERVAL_ENV, _HISTORY_INTERVAL_DEFAULT_S)


def detail_retention_s() -> int:
    return int(_env_float(_DETAIL_RETENTION_ENV, _DETAIL_RETENTION_DEFAULT_S))


def history_retention_s() -> int:
    return int(_env_float(_HISTORY_RETENTION_ENV, _HISTORY_RETENTION_DEFAULT_S))


# ── throttle index (mask → 0-100 severity) ────────────────────────────────
#
# The NVML throttle-reason mask is the smoking-gun field (the GB10 incident
# box died seconds after asserting 0x48) but a hex string doesn't graph.
# ``throttle_index`` collapses it to a 0-100 severity — max over active bits,
# because ONE hw-thermal bit already means "at the wall"; summing would
# overweight benign combinations. ``throttle_reasons`` names the active bits
# for tooltips/logs. Bit values are the stable public NVML constants
# (nvmlClocksThrottleReasons*).

_THROTTLE_BITS: tuple[tuple[int, str, int], ...] = (
    (0x0000000000000001, "gpu_idle", 0),
    (0x0000000000000002, "applications_clocks_setting", 5),
    (0x0000000000000004, "sw_power_cap", 30),
    (0x0000000000000008, "hw_slowdown", 85),
    (0x0000000000000010, "sync_boost", 5),
    (0x0000000000000020, "sw_thermal", 50),
    (0x0000000000000040, "hw_thermal", 100),
    (0x0000000000000080, "hw_power_brake", 90),
    (0x0000000000000100, "display_clock_setting", 5),
)


def _parse_throttle_mask(mask_hex: str | None) -> int | None:
    if mask_hex is None:
        return None
    raw = str(mask_hex).strip()
    if not raw:
        return None
    try:
        return int(raw, 16)
    except (TypeError, ValueError):
        return None  # "[N/A]" on parts that don't report the mask


def throttle_index(mask_hex: str | None) -> int | None:
    """0-100 throttling severity for one GPU's throttle-reason mask —
    ``None`` when the mask is unknown/unreported (a graph gap, not a zero)."""
    mask = _parse_throttle_mask(mask_hex)
    if mask is None:
        return None
    return max(
        (weight for bit, _name, weight in _THROTTLE_BITS if mask & bit),
        default=0,
    )


def throttle_reasons(mask_hex: str | None) -> list[str]:
    """Names of the active throttle bits, severity-relevant ones included —
    empty for no-mask / idle-only / unparseable input."""
    mask = _parse_throttle_mask(mask_hex)
    if mask is None:
        return []
    return [
        name for bit, name, weight in _THROTTLE_BITS
        if mask & bit and weight > 0
    ]


_BRAKE_BITS = 0x88     # hw_slowdown (incl. external power-brake assertion) | hw_power_brake
_THERMAL_BITS = 0x60   # sw_thermal | hw_thermal


def power_brake(mask_hex: str | None) -> bool | None:
    """True when a hardware BRAKE bit is asserted (0x08 hw_slowdown — which
    NVML documents as thermal OR *external power brake assertion* — or 0x80
    hw_power_brake). The GPU-side echo of a power-delivery event; ``None``
    when the mask is unknown."""
    mask = _parse_throttle_mask(mask_hex)
    if mask is None:
        return None
    return bool(mask & _BRAKE_BITS)


def power_brake_pure(mask_hex: str | None) -> bool | None:
    """True when a brake bit fires with NO thermal bit alongside — the
    strongest software-visible hint of a power-side (adapter/VRM) event, as
    opposed to a thermal one wearing the brake bit."""
    mask = _parse_throttle_mask(mask_hex)
    if mask is None:
        return None
    return bool(mask & _BRAKE_BITS) and not (mask & _THERMAL_BITS)


# ── deep sample (every probe best-effort; stable key set) ─────────────────
#
# Vendor GPU probe is picked once per process (nvidia-smi, else rocm-smi, else
# none) — the hw_metrics pattern. All /sys and /proc readers are plain file
# reads; on a non-Linux host they simply return empty.

_DEEP_GPU_PROBE: Callable[[], list[dict[str, Any]]] | None = None


def _num(v: str) -> float | int | None:
    """Permissive numeric parse — smi emits ``[N/A]`` / ``N/A`` on unified parts."""
    v = v.strip()
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return None


def _nvidia_deep() -> list[dict[str, Any]]:
    # power.draw.instant rides along with the averaged power.draw: the SPREAD
    # between them is the transient-amplitude proxy for power-delivery stress
    # (GB10 exposes no input-rail telemetry at all — the brick is software-
    # invisible, so GPU-side echoes are the only PSU watchdog signals).
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,power.draw,power.draw.instant,"
            "clocks.sm,utilization.gpu,utilization.memory,pstate,"
            "clocks_throttle_reasons.active,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        stderr=subprocess.DEVNULL, timeout=2,
    ).decode()
    out: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 10:
            continue
        out.append({
            "temp_c": _num(parts[0]),
            "power_w": _num(parts[1]),
            "power_inst_w": _num(parts[2]),
            "sm_mhz": _num(parts[3]),
            "util_pct": _num(parts[4]),
            "mem_util_pct": _num(parts[5]),
            "pstate": parts[6] or None,
            "throttle_hex": parts[7] or None,
            "vram_used_mb": _num(parts[8]),
            "vram_total_mb": _num(parts[9]),
        })
    return out


def _rocm_deep() -> list[dict[str, Any]]:
    raw = subprocess.check_output(
        ["rocm-smi", "--json", "--showtemp", "--showpower", "--showuse"],
        stderr=subprocess.DEVNULL, timeout=2,
    ).decode()
    data = json.loads(raw)
    out: list[dict[str, Any]] = []
    for key in sorted(k for k in data if k.startswith("card")):
        val = data[key] or {}
        temp = power = use = None
        for k, v in val.items():
            lk = k.lower()
            if temp is None and "temperature" in lk:
                temp = _num(str(v))
            elif power is None and "power" in lk:
                power = _num(str(v))
            elif use is None and "gpu use" in lk:
                use = _num(str(v).rstrip("%"))
        out.append({
            "temp_c": temp, "power_w": power, "util_pct": use,
            "sm_mhz": None, "mem_util_pct": None, "pstate": None,
            "throttle_hex": None, "vram_used_mb": None, "vram_total_mb": None,
        })
    return out


#: nvidia-smi "Violations" labels → our snapshot keys. These are MONOTONIC
#: accumulated-microsecond counters (since driver load) for time the GPU spent
#: throttling for each cause — a strictly better signal than sampling the
#: instantaneous mask (nothing sub-sample is missed; the delta between two
#: samples is the exact throttle time in that window). The POWER-side pair
#: (sw_power_cap + hw_power_brake) is the software-visible proxy for
#: power-delivery stress on a platform that exposes no input-rail telemetry.
_VIOLATION_LABELS = {
    "SW Power Capping": "sw_power_us",
    "HW Power Braking": "hw_power_brake_us",
    "SW Thermal Slowdown": "sw_thermal_us",
    "HW Thermal Slowdown": "hw_thermal_us",
}
_VIOLATION_RE = re.compile(
    r"^\s*(SW Power Capping|HW Power Braking|SW Thermal Slowdown|"
    r"HW Thermal Slowdown)\s*:\s*(\d+)\s*us\s*$"
)


def _gpu_violations() -> dict[str, int]:
    """Accumulated per-cause throttle time (µs) from ``nvidia-smi -q -d
    PERFORMANCE`` → the ``Violations`` block. The ``us`` suffix distinguishes
    these counters from the same-named Active/Not-Active clock-event lines.
    Best-effort: ``{}`` on any failure (non-nvidia box, parse miss). Single
    GPU on this fleet, so the counters are box-wide."""
    raw = subprocess.check_output(
        ["nvidia-smi", "-q", "-d", "PERFORMANCE"],
        stderr=subprocess.DEVNULL, timeout=3,
    ).decode()
    out: dict[str, int] = {}
    for line in raw.splitlines():
        m = _VIOLATION_RE.match(line)
        if m:
            out[_VIOLATION_LABELS[m.group(1)]] = int(m.group(2))
    return out


def _select_deep_gpu_probe() -> Callable[[], list[dict[str, Any]]]:
    for cmd, probe in (("nvidia-smi", _nvidia_deep), ("rocm-smi", _rocm_deep)):
        try:
            subprocess.check_call(
                ["which", cmd],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return probe
        except Exception:  # noqa: BLE001 — probe selection is best-effort
            continue
    return lambda: []


def _gpu_deep() -> list[dict[str, Any]]:
    global _DEEP_GPU_PROBE
    if _DEEP_GPU_PROBE is None:
        _DEEP_GPU_PROBE = _select_deep_gpu_probe()
    return _DEEP_GPU_PROBE()


def _read_thermal_zones() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try:
            with open(os.path.join(zone, "type")) as f:
                ztype = f.read().strip()
            with open(os.path.join(zone, "temp")) as f:
                milli_c = int(f.read().strip())
        except (OSError, ValueError):
            continue
        out.append({"type": ztype, "milli_c": milli_c})
    return out


def _read_hwmon_temps() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for hw in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            with open(os.path.join(hw, "name")) as f:
                name = f.read().strip()
        except OSError:
            continue
        temps: list[int] = []
        for t in sorted(glob.glob(os.path.join(hw, "temp*_input"))):
            try:
                with open(t) as f:
                    temps.append(int(f.read().strip()))
            except (OSError, ValueError):
                continue
        if temps:
            out.append({"name": name, "temps_milli_c": temps})
    return out


def _read_meminfo() -> dict[str, int]:
    wanted = {
        "MemTotal": "total_kb", "MemFree": "free_kb",
        "MemAvailable": "available_kb",
        "SwapTotal": "swap_total_kb", "SwapFree": "swap_free_kb",
    }
    out: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key = line.split(":", 1)[0]
            if key in wanted:
                out[wanted[key]] = int(line.split()[1])
    return out


def _read_load1() -> float | None:
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def _disk_root_used_mb() -> int | None:
    st = os.statvfs("/")
    used = (st.f_blocks - st.f_bfree) * st.f_frsize
    return int(used // (1024 * 1024))


def deep_sample() -> dict[str, Any]:
    """One full hardware sample. Every probe is individually guarded so the
    key set is stable even when a probe fails — consumers can rely on the
    shape, and a broken sensor can never raise into the recorder."""
    sample: dict[str, Any] = {}
    for key, probe, empty in (
        ("gpu", _gpu_deep, []),
        ("tz", _read_thermal_zones, []),
        ("hwmon", _read_hwmon_temps, []),
        ("mem", _read_meminfo, {}),
        ("load1", _read_load1, None),
        ("disk_root_used_mb", _disk_root_used_mb, None),
    ):
        try:
            sample[key] = probe()
        except Exception:  # noqa: BLE001 — a dead probe records its empty shape
            sample[key] = empty
    # Derived, graph-ready: per-GPU 0-100 throttle severity + the box-level
    # worst (what a fleet chart plots). Computed at sample time so readers
    # (the panel reads the DB directly, no engine import) never re-decode.
    indices: list[int] = []
    for g in sample["gpu"]:
        idx = throttle_index(g.get("throttle_hex"))
        g["throttle_index"] = idx
        g["throttle_reasons"] = throttle_reasons(g.get("throttle_hex"))
        if idx is not None:
            indices.append(idx)
    sample["throttle_index"] = max(indices) if indices else None
    # Accumulated throttle-time counters (µs, monotonic) — the exact
    # power-vs-thermal throttle breakdown; attached to the first GPU (single
    # GPU per box on this fleet). Own guard so a parse miss can't drop the rest.
    if sample["gpu"]:
        try:
            vio = _gpu_violations()
        except Exception:  # noqa: BLE001 — best-effort, like every probe
            vio = {}
        if vio:
            sample["gpu"][0]["violations"] = vio
    return sample


# ── storage: record / read / prune ───────────────────────────────────────


def _jsonb(value: dict[str, Any] | None):
    """psycopg Jsonb shortcut (the sqlite adapter handles it too — see
    ``db._register_sqlite_adapters``)."""
    from psycopg.types.json import Jsonb
    return Jsonb(value or {})


def record_hw_sample(
    *,
    host_label: str,
    tier: str,
    data: dict[str, Any],
    box: str | None = None,
    project: str | None = None,
    created_at: datetime | None = None,
) -> int | None:
    """Best-effort append of one sample on its OWN connection.

    Swallows EVERY exception (logs once) and returns ``None`` on failure — a
    telemetry blip, a pre-0021 schema, or a dead DB can never take down the
    sampling worker. ``created_at`` is an injection seam for retention tests;
    live callers let the column default stamp it. Returns 1 on success."""
    if project is None:
        from queue_workflows.config import get_config
        project = get_config().project
    try:
        with connection() as conn, conn.cursor() as cur:
            if created_at is None:
                cur.execute(
                    "INSERT INTO hw_watch_samples"
                    " (host_label, box, project, tier, data)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    (host_label, box, project or "", tier, _jsonb(data)),
                )
            else:
                cur.execute(
                    "INSERT INTO hw_watch_samples"
                    " (host_label, box, project, tier, data, created_at)"
                    " VALUES (%s, %s, %s, %s, %s, %s)",
                    (host_label, box, project or "", tier, _jsonb(data),
                     created_at),
                )
            return 1
    except Exception:  # noqa: BLE001 — best-effort; never propagate
        log.exception(
            "[hw-watch] failed to record %s sample for %s (ignored)",
            tier, host_label,
        )
        return None


def recent_hw_samples(
    *,
    host_label: str | None = None,
    tier: str = TIER_DETAIL,
    since_s: int = 3600,
    limit: int = 2000,
) -> list[dict[str, Any]]:
    """Read back a recent trail, newest first, ``data`` decoded to a dict on
    both backends (sqlite stores JSON TEXT)."""
    d = get_dialect()
    sql = (
        "SELECT * FROM hw_watch_samples"
        f" WHERE tier = %s AND created_at >= {d.past_seconds('%s')}"
    )
    params: list[Any] = [tier, int(since_s)]
    if host_label is not None:
        sql += " AND host_label = %s"
        params.append(host_label)
    sql += " ORDER BY created_at DESC LIMIT %s"
    params.append(int(limit))
    with connection() as conn, conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        rows = [dict(r) for r in cur.fetchall()]
    for row in rows:
        if isinstance(row.get("data"), str):
            try:
                row["data"] = json.loads(row["data"])
            except (TypeError, ValueError):
                pass
    return rows


def prune_hw_watch(
    *,
    detail_retention: int | None = None,
    history_retention: int | None = None,
) -> tuple[int, int]:
    """Two-tier retention sweep: delete ``detail`` rows older than 1 h and
    ``history`` rows older than 24 h (env-tunable). Returns
    ``(detail_deleted, history_deleted)``. Swallows storage errors — a
    pre-0021 DB (or a sweep-time blip) prunes nothing rather than spamming
    the orchestrator log with tracebacks."""
    detail_s = detail_retention if detail_retention is not None else detail_retention_s()
    history_s = history_retention if history_retention is not None else history_retention_s()
    d = get_dialect()
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM hw_watch_samples"
                f" WHERE tier = %s AND created_at < {d.past_seconds('%s')}",
                (TIER_DETAIL, int(detail_s)),
            )
            detail_deleted = cur.rowcount or 0
            cur.execute(
                "DELETE FROM hw_watch_samples"
                f" WHERE tier = %s AND created_at < {d.past_seconds('%s')}",
                (TIER_HISTORY, int(history_s)),
            )
            history_deleted = cur.rowcount or 0
            return detail_deleted, history_deleted
    except Exception:  # noqa: BLE001 — sweep is best-effort (pre-0021 DBs)
        log.debug("[hw-watch] prune skipped (storage error)", exc_info=True)
        return 0, 0


# ── cadence brain (pure; virtual-clock testable) ─────────────────────────

# Module-level aliases so ``HwWatchRecorder.__init__`` can reach the env-knob
# functions past its same-named keyword parameters.
_detail_interval_default = detail_interval_s
_history_interval_default = history_interval_s


def _default_box() -> str | None:
    try:
        from queue_workflows.gpu_model_lease import default_box_id
        return default_box_id()
    except Exception:  # noqa: BLE001 — box identity is optional metadata
        return None


class HwWatchRecorder:
    """Decides *when* each tier is due and writes ONE shared deep sample for
    all tiers due that tick. Pure logic — inject ``now_fn`` / ``record_fn`` /
    ``sample_fn`` to test on a virtual clock. The first tick records both
    tiers immediately (a fresh worker leaves a sample right away — the box
    may not live long, as the motivating incident proved)."""

    def __init__(
        self,
        *,
        host_label: str,
        box: str | None = None,
        project: str | None = None,
        detail_interval_s: float | None = None,
        history_interval_s: float | None = None,
        now_fn: Callable[[], float] = time.time,
        record_fn: Callable[..., int | None] | None = None,
        sample_fn: Callable[[], dict[str, Any]] | None = None,
    ):
        self.host_label = host_label
        self.box = box if box is not None else _default_box()
        self.project = project
        self.detail_interval_s = (
            float(detail_interval_s) if detail_interval_s is not None
            else _detail_interval_default()
        )
        self.history_interval_s = (
            float(history_interval_s) if history_interval_s is not None
            else _history_interval_default()
        )
        self._now_fn = now_fn
        self._record_fn = record_fn or record_hw_sample
        self._sample_fn = sample_fn or deep_sample
        self._next_detail = 0.0
        self._next_history = 0.0

    def tick(self, now: float | None = None) -> list[str]:
        """Record every tier that is due at ``now``. Returns the tiers
        written (possibly empty). The deep sample runs at most ONCE per tick
        and only when something is due."""
        t = self._now_fn() if now is None else now
        due: list[str] = []
        if t >= self._next_detail:
            due.append(TIER_DETAIL)
            self._next_detail = t + self.detail_interval_s
        if t >= self._next_history:
            due.append(TIER_HISTORY)
            self._next_history = t + self.history_interval_s
        if not due:
            return []
        data = self._sample_fn()
        for tier in due:
            self._record_fn(
                host_label=self.host_label, tier=tier, data=data,
                box=self.box, project=self.project,
            )
        return due


# ── standalone CLI (queue-hw-watch) ──────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    """Standalone flight recorder for one box — no claim worker needed (e.g.
    a machine under investigation with its workers parked). Bootstraps the
    engine chain idempotently (the ``queue-broker`` standalone precedent),
    then records forever and prunes its own retention every 5 min."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--db-backend", choices=("pg", "sqlite"), default=None,
                        help="override QUEUE_WORKFLOWS_DB_BACKEND")
    parser.add_argument("--host-label", default=None,
                        help="override the emitting label (default: hw_metrics host label)")
    parser.add_argument("--detail-interval", type=float, default=None)
    parser.add_argument("--history-interval", type=float, default=None)
    parser.add_argument("--once", action="store_true",
                        help="record one sample per tier and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.db_backend:
        from queue_workflows import configure
        configure(db_backend=args.db_backend)

    from queue_workflows import db
    db.bootstrap()

    if args.host_label:
        host_label = args.host_label
    else:
        from queue_workflows.hw_metrics import _host_label
        host_label = _host_label()

    recorder = HwWatchRecorder(
        host_label=host_label,
        detail_interval_s=args.detail_interval,
        history_interval_s=args.history_interval,
    )
    log.info(
        "[hw-watch] recording %s (box=%s) detail=%ss/1h history=%ss/24h",
        host_label, recorder.box,
        recorder.detail_interval_s, recorder.history_interval_s,
    )
    if args.once:
        recorder.tick()
        return
    next_prune = 0.0
    while True:
        recorder.tick()
        now = time.time()
        if now >= next_prune:
            prune_hw_watch()
            next_prune = now + 300.0
        time.sleep(min(recorder.detail_interval_s, 5.0))


if __name__ == "__main__":  # pragma: no cover
    main()
