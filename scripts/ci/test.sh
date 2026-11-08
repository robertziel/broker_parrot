#!/usr/bin/env bash
# broker_parrot's CI gates — the same on Jenkins and on any machine with Docker.
#
#   scripts/ci/test.sh sqlite   the whole suite on the hermetic SQLite backend (QUEUE_WORKFLOWS_TEST_SQLITE=1)
#   scripts/ci/test.sh pg       the whole suite against a throwaway postgres container (torn down after)
#   scripts/ci/test.sh all      both, in that order
#   scripts/ci/test.sh image    only build the suite image (Jenkins does this once, before the two gates run in parallel)
#
# The checkout is mounted READ-ONLY and copied inside the test container, so nothing here writes to it
# (a Jenkins workspace stays deletable, a dev checkout keeps its own .venv / caches untouched).
set -euo pipefail

cd "$(dirname "$0")/../.."
SRC=$PWD
PY_IMAGE=${CI_PY_IMAGE:-python:3.13-slim}
PG_IMAGE=${CI_PG_IMAGE:-postgres:16-alpine}

# Bind mounts are resolved by the Docker DAEMON. Under Jenkins this script runs inside the controller's container,
# talking to the host's daemon, so it needs the checkout's path on the HOST: the source of the container mount that
# holds $SRC. Not in a container → $SRC itself. Inside one with no such mount → fail; never guess a path.
host_path() {
  if [ ! -f /.dockerenv ]; then echo "$SRC"; return; fi
  local dest source
  while IFS='|' read -r dest source; do
    [ -n "$dest" ] || continue
    case "$SRC/" in "$dest"/*) echo "$source${SRC#"$dest"}"; return ;; esac
  done < <(docker inspect "$(hostname)" --format '{{range .Mounts}}{{.Destination}}|{{.Source}}{{"\n"}}{{end}}')
  echo "ci: $SRC is not on any mount of container $(hostname) — cannot hand it to the Docker daemon" >&2
  exit 1
}
HOST_SRC=$(host_path)

# One name per run AND per gate: concurrent builds, and the sqlite + pg gates Jenkins runs in parallel
# within one build (same BUILD_TAG), must never share a container name.
RUN_ID=$(printf '%s' "${BUILD_TAG:-local-$$}" | tr -c 'a-zA-Z0-9_-' '-' | tr 'A-Z' 'a-z')
NAME="broker-parrot-ci-${RUN_ID}-${1:-none}"

# An aborted or timed-out build must not leave containers behind: remove this run's suite + db containers on any exit.
cleanup() { docker rm -f "$NAME-suite" "$NAME-pg" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

# pytest-xdist workers per suite: every CPU the suite container may run on (its scheduler affinity — the number
# `-n auto` picks; the host's raw core count over-counts when cores are reserved); CI_PYTEST_WORKERS=N overrides it
# (0 = serial). Each worker gets its own database / SQLite file (tests/conftest.py). Resolved HERE, not left to
# `auto`, because the pg gate sizes max_connections from it.
# (`image` runs no suite: it skips the probe container.)
if [ "${1:-}" = image ]; then PYTEST_WORKERS=0; else
  PYTEST_WORKERS=${CI_PYTEST_WORKERS:-$(docker run --rm "$PY_IMAGE" python -c 'import os; print(len(os.sched_getaffinity(0)))')}
fi
case "$PYTEST_WORKERS" in ''|*[!0-9]*) echo "ci: worker count '$PYTEST_WORKERS' is not a number" >&2; exit 1 ;; esac
# Postgres connection budget: each worker process holds up to DB_POOL_MAX pooled connections (pinned for the suite,
# so this script and the engine read the same number) plus a few direct ones (LISTEN, the maintenance DB).
DB_POOL_MAX=10
PG_MAX_CONNECTIONS=$(( (PYTEST_WORKERS > 0 ? PYTEST_WORKERS : 1) * (DB_POOL_MAX + 5) + 20 ))

# The suite image (scripts/ci/Dockerfile): $PY_IMAGE + git + every [test] dependency, tagged by a hash of what it is
# built from, so it is built once per dependency change and every other run reuses it (no apt / pip download per run).
# The build context is only those two files, streamed from here — no bind mount, so no host path is needed.
CI_IMAGE="broker-parrot-ci:$( { cat scripts/ci/Dockerfile pyproject.toml; printf '%s' "$PY_IMAGE"; } | sha256sum | cut -c1-16)"
build_image() {
  tar cf - scripts/ci/Dockerfile pyproject.toml \
    | docker build -q --build-arg PY_IMAGE="$PY_IMAGE" -f scripts/ci/Dockerfile -t "$CI_IMAGE" - >/dev/null
  echo "ci: suite image $CI_IMAGE"
}

# run pytest inside $CI_IMAGE on a copy of the read-only checkout; extra docker args come first. The package itself is
# installed --no-deps with no build isolation: everything it needs is already in the image, so this step is offline.
run_suite() {
  build_image
  docker run --rm --name "$NAME-suite" "$@" -e PYTEST_WORKERS="$PYTEST_WORKERS" -e QUEUE_WORKFLOWS_DB_POOL_MAX="$DB_POOL_MAX" -v "$HOST_SRC:/src:ro" "$CI_IMAGE" sh -euc '
    mkdir /w && cd /src && tar cf - --exclude=./.venv --exclude=./__pycache__ --exclude=./.pytest_cache . | tar xf - -C /w
    cd /w && pip install -q --no-deps --no-build-isolation -e .
    python -m pytest -q -p no:cacheprovider -n "$PYTEST_WORKERS"'
}

sqlite_suite() {
  run_suite -e QUEUE_WORKFLOWS_TEST_SQLITE=1
}

pg_suite() {
  local status=0
  # no user network (a busy daemon can run out of address pools): the suite joins the db container's own
  # network namespace and reaches it on 127.0.0.1
  # a throwaway db: data on tmpfs and no fsync. Every test TRUNCATEs the engine tables, and on a slow disk each
  # TRUNCATE's sync turns a 2-minute suite into a 30-minute one (CI build timeout)
  docker run -d --rm --name "$NAME-pg" --tmpfs /var/lib/postgresql/data \
    -e POSTGRES_USER=ci -e POSTGRES_PASSWORD=ci -e POSTGRES_DB=queue_workflows_test "$PG_IMAGE" \
    -c fsync=off -c synchronous_commit=off -c full_page_writes=off -c max_connections="$PG_MAX_CONNECTIONS" >/dev/null
  # wait for the server (it restarts once after initdb, so require a real query, not just pg_isready)
  local i
  for i in $(seq 1 60); do
    docker exec "$NAME-pg" psql -U ci -d queue_workflows_test -tAc 'select 1' >/dev/null 2>&1 && break
    [ "$i" = 60 ] && { echo "ci: postgres did not come up in 60 s" >&2; status=1; }
    sleep 1
  done
  if [ "$status" = 0 ]; then
    run_suite --network "container:$NAME-pg" \
      -e QUEUE_WORKFLOWS_TEST_DB_URL="postgresql://ci:ci@127.0.0.1:5432/queue_workflows_test" || status=$?
  fi
  cleanup
  return $status
}

case "${1:-}" in
  sqlite) sqlite_suite ;;
  pg) pg_suite ;;
  all) sqlite_suite && pg_suite ;;
  image) build_image ;;
  *) echo "usage: $0 sqlite|pg|all|image" >&2; exit 2 ;;
esac
