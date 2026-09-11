#!/usr/bin/env bash
# hw-broadcast.sh -- run THIS checkout's hw-metrics sampler with LAN broadcast on.
#
# For a host that wants to feed a LAN display panel but runs no engine claim
# worker. Where a claim worker already runs, prefer simply setting
# QUEUE_WORKFLOWS_BROADCAST_HW_STATS=1 on it -- its sampler already does this,
# and one sampler per host is the design.
#
#   --network host   REQUIRED: a bridged container's broadcasts die in its own
#                    netns, and interface discovery would see veth, not the LAN.
#   --restart        boot persistence, without a systemd unit or sudo.
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${HW_BROADCAST_NAME:-hw-broadcast}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
# GPU telemetry shells out to rocm-smi / nvidia-smi. A slim container has
# neither, so the GPU column would be empty; mount the host's ROCm tree and the
# render nodes when they exist. (On a real fleet host this script is not needed
# at all -- the sampler runs inside the gpu claim worker, which already has the
# CLI and the devices.)
GPU_ARGS=()
if [ -d /opt/rocm ]; then
    # AMD: mount the ROCm tree so rocm-smi is on PATH inside the container.
    GPU_ARGS+=(-v /opt/rocm:/opt/rocm:ro -e PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin)
    [ -e /dev/kfd ] && GPU_ARGS+=(--device /dev/kfd)
    [ -e /dev/dri ] && GPU_ARGS+=(--device /dev/dri)
    [ -e /sys/class/drm ] && GPU_ARGS+=(-v /sys:/sys:ro)
elif command -v nvidia-smi >/dev/null 2>&1; then
    # NVIDIA: prefer the container toolkit; it puts nvidia-smi in the container
    # for us. Without it the GPU row is simply empty -- CPU/RAM still flow, so
    # this must never be fatal.
    if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
        GPU_ARGS+=(--gpus all)
    else
        echo "note: nvidia-smi present but no nvidia docker runtime; GPU row will be empty" >&2
    fi
fi

docker run -d --name "$NAME" --restart unless-stopped --network host \
    "${GPU_ARGS[@]}" \
    -v "$ROOT:/src:ro" -w /src \
    -e QUEUE_WORKFLOWS_BROADCAST_HW_STATS=1 \
    -e QUEUE_WORKFLOWS_DB_BACKEND=sqlite \
    -e QUEUE_WORKFLOWS_DB_URL=/tmp/hw-broadcast.sqlite \
    -e PYTHONPATH=/src \
    python:3.12-slim sh -c 'pip install --no-cache-dir --quiet psutil || true; exec python /src/deploy/hw_broadcast_main.py'

sleep 10
docker logs "$NAME" 2>&1 | tail -4
echo "follow with:  docker logs -f $NAME"
