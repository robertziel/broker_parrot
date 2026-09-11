"""Entrypoint for deploy/hw-broadcast.sh — sample this host, broadcast on the LAN.

Deliberately thin: it owns no sampling logic of its own. The engine's
``hw_metrics._build_sample`` does the probing and ``hw_lan.LanBroadcaster`` does
the sending, exactly as the in-process sampler does — this only supplies the
loop for a host that is not yet running an engine claim worker with
``broadcast_hw_stats`` enabled.
"""

from __future__ import annotations

import logging
import time

from queue_workflows import hw_lan, hw_metrics

INTERVAL_S = 5.0

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hw-broadcast")


def main() -> None:
    log.info("broadcasting to %s on port %s", hw_lan.discover_targets(), hw_lan.port())
    if not hw_lan.enabled():
        log.error("broadcast_hw_stats is OFF — set "
                  "QUEUE_WORKFLOWS_BROADCAST_HW_STATS=1; nothing will be sent")
    broadcaster = hw_lan.LanBroadcaster()
    sent = 0
    while True:
        try:
            if broadcaster.publish(hw_metrics._build_sample()):
                sent += 1
                if sent % 60 == 0:
                    log.info("published %d samples", sent)
        except Exception:
            log.exception("publish failed")
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
