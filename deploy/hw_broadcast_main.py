"""Entrypoint for deploy/hw-broadcast.sh — sample this host, broadcast on the LAN.

Deliberately thin: it owns no sampling logic of its own. The engine's
``hw_metrics._build_sample`` does the probing and ``hw_lan.LanBroadcaster`` does
the sending, exactly as the in-process sampler does — this only supplies the
loop for a host that is not yet running an engine claim worker with
``broadcast_hw_stats`` enabled.

One ADDITIVE field is added here, not in the engine sample: ``cpu_per_core``
(a list of per-core CPU %, psutil order), for a receiver's per-core load matrix.
Receivers that predate it ignore unknown keys; the NOTIFY path is untouched.
"""

from __future__ import annotations

import logging
import time

from queue_workflows import hw_lan, hw_metrics

INTERVAL_S = 5.0

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hw-broadcast")


def _cpu_per_core() -> list[float] | None:
    """Per-core CPU %, the same non-blocking "since the last call" reading that
    ``_build_sample`` takes for ``cpu_percent``. psutil keeps the per-core
    baseline apart from the system-wide one, so this never disturbs
    ``cpu_percent``. ``None`` without psutil or on any failure — never fatal."""
    psutil = hw_metrics.psutil
    if psutil is None:
        return None
    try:
        return [float(v) for v in psutil.cpu_percent(interval=None, percpu=True)]
    except Exception:
        log.debug("per-core cpu probe failed", exc_info=True)
        return None


def main() -> None:
    log.info("broadcasting to %s on port %s", hw_lan.discover_targets(), hw_lan.port())
    if not hw_lan.enabled():
        log.error("broadcast_hw_stats is OFF — set "
                  "QUEUE_WORKFLOWS_BROADCAST_HW_STATS=1; nothing will be sent")
    broadcaster = hw_lan.LanBroadcaster()
    sent = 0
    while True:
        try:
            # Read the per-core figures immediately before _build_sample reads
            # cpu_percent, so both cover the same since-last-call window.
            per_core = _cpu_per_core()
            sample = hw_metrics._build_sample()
            sample["cpu_per_core"] = per_core
            if broadcaster.publish(sample):
                sent += 1
                if sent % 60 == 0:
                    log.info("published %d samples", sent)
        except Exception:
            log.exception("publish failed")
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
