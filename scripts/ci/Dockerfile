# The CI suite image: the Python image + git + every dependency of the [test] extra + the build backend, so a CI run
# only copies the checkout and installs the package itself (--no-deps, offline). Built by scripts/ci/test.sh and
# tagged with a hash of this file + pyproject.toml + PY_IMAGE, so it is rebuilt exactly when the dependencies change.
ARG PY_IMAGE
FROM ${PY_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore
RUN apt-get -qq update >/dev/null && apt-get -qq install -y --no-install-recommends git >/dev/null \
 && rm -rf /var/lib/apt/lists/* && git config --global --add safe.directory /w
COPY pyproject.toml /deps/pyproject.toml
# the dependency list straight from pyproject (runtime + [test]), plus hatchling/editables for the --no-build-isolation
# editable install a run does
RUN python -c "import tomllib; p = tomllib.load(open('/deps/pyproject.toml', 'rb'))['project']; \
print('\n'.join(p['dependencies'] + p['optional-dependencies']['test'] + ['hatchling', 'editables']))" > /deps/requirements.txt \
 && pip install -q -r /deps/requirements.txt
