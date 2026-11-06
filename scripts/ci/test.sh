#!/usr/bin/env bash
# broker_parrot's CI gates — the same on Jenkins and on any machine with Docker.
#
#   scripts/ci/test.sh sqlite   the whole suite on the hermetic SQLite backend (QUEUE_WORKFLOWS_TEST_SQLITE=1)
#   scripts/ci/test.sh pg       the whole suite against a throwaway postgres container (torn down after)
#   scripts/ci/test.sh all      both, in that order
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

# install + run pytest inside $PY_IMAGE on a copy of the read-only checkout; extra docker args come first
run_suite() {
  docker run --rm --name "$NAME-suite" "$@" -v "$HOST_SRC:/src:ro" "$PY_IMAGE" sh -euc '
    apt-get -qq update >/dev/null && apt-get -qq install -y --no-install-recommends git >/dev/null
    mkdir /w && cd /src && tar cf - --exclude=./.venv --exclude=./__pycache__ --exclude=./.pytest_cache . | tar xf - -C /w
    cd /w && git config --global --add safe.directory /w
    pip install -q --disable-pip-version-check --root-user-action=ignore -e ".[test]"
    python -m pytest -q -p no:cacheprovider'
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
    -c fsync=off -c synchronous_commit=off -c full_page_writes=off >/dev/null
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
  *) echo "usage: $0 sqlite|pg|all" >&2; exit 2 ;;
esac
