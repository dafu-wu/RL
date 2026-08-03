#!/bin/bash
# Build the NeMo-RL image that bakes the current repo source into /opt/nemo-rl
# (local disk) to avoid slow network-FS cold imports of Ray workers.
#
# Usage:
#   bash ci/build.sh                                   # uses default IMAGE tag (date-stamped)
#   IMAGE=registry.example.com/nemo-rl:v0.6.0-20260623 bash ci/build.sh
#   PUSH=1 IMAGE=registry.example.com/nemo-rl:v0.6.0-20260623 bash ci/build.sh
#   # Optional auto-login before push (avoids 'unauthorized' from stale creds):
#   PUSH=1 REGISTRY_USER=admin REGISTRY_PASS=*** IMAGE=registry.zoomdev.us/languagetech/nemo-rl:v0.6.0-20260623 bash ci/build.sh
#
# Run this from the RL repo root (the directory that contains pyproject.toml and 3rdparty/).
set -euo pipefail

# ---- config (override via env) ----
# Default image tag is date-stamped (YYYYMMDD), e.g. nemo-rl:v0.6.0-20260623
DATE_TAG="$(date +%Y%m%d)"
IMAGE="${IMAGE:-nemo-rl:v0.6.0-${DATE_TAG}}"   # full <registry>/<name>:<tag>
PUSH="${PUSH:-0}"                              # set PUSH=1 to docker push after build
# Optional credentials for auto-login before push (leave empty to skip auto-login).
REGISTRY_USER="${REGISTRY_USER:-}"
REGISTRY_PASS="${REGISTRY_PASS:-}"
# Target platform(s). Single platform -> plain `docker build` (loads into local docker).
# Multi-platform (e.g. "linux/amd64,linux/arm64") -> buildx with QEMU; requires PUSH=1
# because a manifest-list cannot be loaded into the local docker image store.
PLATFORMS="${PLATFORMS:-}"
# -----------------------------------




# Resolve repo root = parent dir of this script's dir (ci/ -> repo root).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)"
cd "${REPO_ROOT}"

echo "==> Repo root: ${REPO_ROOT}"
echo "==> Image tag: ${IMAGE}"

# 1) Make sure submodules are initialized (their source must be in the build context,
#    because /opt/nemo-rl's editable finders import megatron.core/bridge/automodel/gym
#    from 3rdparty/*/ on local disk).
echo "==> Checking submodules..."
git submodule update --init
MISSING=0
for d in \
    3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core \
    3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge \
    3rdparty/Automodel-workspace/Automodel/nemo_automodel \
    3rdparty/Gym-workspace/Gym/nemo_gym ; do
    if [ ! -d "$d" ]; then
        echo "    MISSING: $d"
        MISSING=1
    fi
done
if [ "$MISSING" != "0" ]; then
    echo "ERROR: some submodule sources are missing; run 'git submodule update --init --recursive' and retry." >&2
    exit 1
fi
echo "    submodules OK"

# 2) Regenerate the container fingerprint from the CURRENT code (pyproject.toml + uv.lock
#    + submodule SHAs). Baked into the image so runtime no longer reports Code/Container
#    mismatch.
echo "==> Generating fingerprint -> ci/nemo_rl_container_fingerprint"
python3 tools/generate_fingerprint.py > ci/nemo_rl_container_fingerprint
cat ci/nemo_rl_container_fingerprint

# Derive the registry host from IMAGE (everything before the first '/').
# If IMAGE has no registry host (e.g. local "nemo-rl:tag"), REGISTRY stays empty.
REGISTRY=""
case "${IMAGE}" in
    */*) host="${IMAGE%%/*}"; case "$host" in *.*|*:*) REGISTRY="$host" ;; esac ;;
esac

do_login() {
    if [ -z "${REGISTRY}" ]; then
        echo "==> No registry host in IMAGE; skip docker login"
        return 0
    fi
    if [ -n "${REGISTRY_USER}" ] && [ -n "${REGISTRY_PASS}" ]; then
        echo "==> docker login ${REGISTRY} as ${REGISTRY_USER}"
        echo "${REGISTRY_PASS}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin
    else
        # No creds provided: refresh existing stored credentials by re-login (uses cred store).
        echo "==> docker login ${REGISTRY} (using stored credentials)"
        docker login "${REGISTRY}" || true
    fi
}

# 3) Build.
if [ -n "${PLATFORMS}" ] && [ "${PLATFORMS}" != "linux/amd64" ]; then
    # ---- Multi-platform build via buildx + QEMU ----
    # A multi-arch manifest list cannot be loaded into the local docker store, so it must be
    # pushed directly to the registry. Hence PUSH=1 is required here.
    if [ "${PUSH}" != "1" ]; then
        echo "ERROR: multi-platform build (PLATFORMS=${PLATFORMS}) requires PUSH=1 (manifest list cannot be --load'ed locally)." >&2
        exit 1
    fi
    echo "==> Multi-platform build: ${PLATFORMS}"

    # Register QEMU emulators (idempotent) so arm64 RUN steps can execute on this amd64 host.
    echo "==> Registering QEMU/binfmt emulators (docker run --privileged tonistiigi/binfmt --install all)"
    docker run --privileged --rm tonistiigi/binfmt --install all >/dev/null 2>&1 || \
        docker run --privileged --rm tonistiigi/binfmt --install all

    # Ensure a buildx builder that supports multiple platforms (docker-container driver).
    BUILDER="nemo-rl-multiarch-builder"
    if ! docker buildx inspect "${BUILDER}" >/dev/null 2>&1; then
        echo "==> Creating buildx builder ${BUILDER}"
        docker buildx create --name "${BUILDER}" --driver docker-container --use >/dev/null
    fi
    docker buildx use "${BUILDER}"
    docker buildx inspect --bootstrap >/dev/null

    # Login before push (multi-arch must push).
    do_login

    echo "==> buildx build+push: ${IMAGE} [${PLATFORMS}]"
    DOCKER_BUILDKIT=1 docker buildx build \
        --platform "${PLATFORMS}" \
        -f ci/Dockerfile \
        -t "${IMAGE}" \
        --push \
        "${REPO_ROOT}"
    echo "==> Multi-platform build+push complete: ${IMAGE} [${PLATFORMS}]"
    echo "==> Done."
    exit 0
fi

# ---- Single-platform build (default): plain docker build, loads into local docker ----
echo "==> Building image (single-arch): ${IMAGE}"
DOCKER_BUILDKIT=1 docker build \
    -f ci/Dockerfile \
    -t "${IMAGE}" \
    "${REPO_ROOT}"

echo "==> Build complete: ${IMAGE}"

# 4) Optional push (with auto-login + one retry to avoid transient 'unauthorized').
if [ "${PUSH}" = "1" ]; then
    echo "==> Pushing ${IMAGE}"
    do_login
    if ! docker push "${IMAGE}"; then
        echo "==> Push failed (possibly stale auth); re-login and retry once..."
        do_login
        docker push "${IMAGE}"
    fi
    echo "==> Push complete: ${IMAGE}"
fi



echo "==> Done. Runtime reminder: set NEMO_RL_HOME=/opt/nemo-rl (and NEMO_RL_PROJECT_ROOT/UV_PROJECT_DIR),"
echo "    run from /opt/nemo-rl, and do NOT mount the network FS over /opt/nemo-rl."
