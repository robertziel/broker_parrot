# LAN broadcast of hardware metrics

An **opt-in, default-off** second publication path for the hw-metrics sample the
engine already collects on every host.

## Why

The engine's `HwMetricsSampler` already probes each box's CPU / RAM / GPU every
5 s and fires `NOTIFY hw_metrics` on the broker database. That serves consumers
that can reach Postgres.

A LAN appliance — a display panel, a status light — is a different kind of
consumer. Giving it a DSN and credentials, or standing up a sidecar container
that re-probes a box the engine is already probing, is all cost and no new
information. Broadcasting the *same* sample on the local network lets such a
listener work with **zero configuration**: no address for the host, no
credentials, survives DHCP changes on either side, and one datagram serves any
number of listeners.

The trade-off is that broadcast does not cross subnets. That is acceptable
because "the local network" *is* the broadcast domain.

## Enabling it

Off unless asked for — a host must opt in before its telemetry goes on the wire.

```python
import queue_workflows
queue_workflows.configure(broadcast_hw_stats=True)
```

or, for entrypoints with no host `configure()` (the console scripts):

```bash
export QUEUE_WORKFLOWS_BROADCAST_HW_STATS=1
export QUEUE_WORKFLOWS_HW_BROADCAST_PORT=48291   # optional; this is the default
```

A retired `AI_LEADS_*` spelling of either knob is a startup error, via `envcompat`.

## What goes on the wire

Exactly the envelope the Postgres NOTIFY carries — same builder
(`hw_metrics.build_envelope`), so the two paths cannot drift:

```json
{"sampled_at": "2026-01-01T12:00:00.000000+00:00", "host": "box-a",
 "cpu_percent": 50.0, "cpu_ours_percent": null,
 "ram_percent": 25.0, "ram_used_mb": 16384, "ram_total_mb": 65536,
 "ram_ours_used_mb": null, "ram_ours_percent": null, "swap_used_mb": 0,
 "gpus": [{"index": 0, "util_percent": 0, "vram_used_mb": 512, ...}]}
```

One UDP datagram per target per sample. Targets default to the **default-route
interface's** broadcast address plus `255.255.255.255`; a box with VPN or
secondary subnets does not spray telemetry at those peers unless asked
(`LanBroadcaster(all_interfaces=True)`). Because both a subnet-directed and a
limited broadcast are sent, a listener normally sees each sample **twice** and
should de-duplicate on `sampled_at` + `host`.

`psutil` (the `[metrics]` extra) is what supplies CPU and RAM; without it those
fields are `null` and only GPU telemetry flows. Install it on any host you want
full panels from.

## Guarantees

- **Off by default.** No socket is created and nothing is sent until enabled.
- **No extra sampling.** The sample handed to the NOTIFY is the one broadcast;
  this adds a transport, not a probe.
- **Never fatal.** Every network error is logged and swallowed — a claim worker
  must not die for want of a status datagram. A full send buffer drops the
  sample rather than stalling the sampler thread; the next one supersedes it.

## Security

The payload is unauthenticated and unencrypted, and it names the host and its
load. Only enable it on a network you trust. Anyone on the broadcast domain can
read it — and, symmetrically, a listener must treat what it receives as
untrusted input.

See `queue_workflows/hw_lan.py` and `tests/test_hw_lan_broadcast.py`.
