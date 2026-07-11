#!/usr/bin/env bash
# broker_parrot — ONE container running N concurrency-1 claim-worker PROCESSES per lane
# (the recommended default; see ../docs/single_container_worker_lane.md). Reference
# supervisor: spawns N processes, restarts a crashed one (replacing docker's per-container
# `restart: unless-stopped`), and forwards SIGTERM to every child so leases drain within the
# compose stop_grace_period. If the whole container dies, docker restarts it.
#
# CLAIM_MODULE = the consumer's claim entrypoint (defaults to the engine's own console
# module); each consumer typically wraps it (e.g. `workflows.workflow_runtime.lib_claim`).
# Usage: run_claim_lane.sh <queue>   (count from LM_WORKFLOW_<QUEUE>_WORKERS)
set -u

QUEUE="${1:?usage: run_claim_lane.sh <queue>}"
CLAIM_MODULE="${CLAIM_MODULE:-queue_workflows.claim_worker}"
case "$QUEUE" in
  cpu) N="${LM_WORKFLOW_CPU_WORKERS:-30}" ;;
  gpu) N="${LM_WORKFLOW_GPU_WORKERS:-1}" ;;
  *)   N="${LM_WORKFLOW_LANE_WORKERS:-1}" ;;
esac
case "$N" in ''|*[!0-9]*) N=1 ;; esac
[ "$N" -lt 1 ] && N=1

pids=()
_term() {
  echo "[claim-lane:$QUEUE] SIGTERM — stopping $N worker(s) for graceful lease drain"
  for p in "${pids[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
  wait
  exit 0
}
trap _term SIGTERM SIGINT

echo "[claim-lane:$QUEUE] starting $N process(es) of $CLAIM_MODULE in one container"
for i in $(seq 1 "$N"); do
  (
    while true; do
      python -m "$CLAIM_MODULE" --queue "$QUEUE"
      echo "[claim-lane:$QUEUE] worker $i exited ($?); restart in 2s"
      sleep 2
    done
  ) &
  pids+=("$!")
done
wait
