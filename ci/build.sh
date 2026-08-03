#!/bin/bash
# Build a SINGLE-ARCH NeMo-RL image (ci/Dockerfile) for the requested arch and LOAD it into
# the local docker image store (no registry push here). We tag it TWICE:
#   1) zoomai/mlp/${MODULE_NAME}          -> the name the Zoom CI publish stage expects
#      (publish runs `docker tag zoomai/mlp/${MODULE_NAME} <registry>/...:<VERSION>` then
#       pushes; if this tag is missing you get "No such image: zoomai/mlp/...:latest").
#   2) zoomai/mlp/${MODULE_NAME}-${ARCH}  -> arch-suffixed tag so amd64/arm64 builds are
#      distinguishable locally and never overwrite each other.
# Style mirrors ml-runtime-env/ci/build.sh and zoom-v2/global_scheduler/ci/build.sh.
#
# The image bakes the current repo source into /opt/nemo-rl (local disk) so Ray
# worker imports read from local disk instead of the network FS. The submodule
# sources under 3rdparty/* are excluded via ci/Dockerfile.dockerignore, so the
# base image's v0.6.0 editable installs are left untouched.
#
# ci/Dockerfile selects the matching Zoom-internal apt source per target arch:
#   - amd64 -> ubuntu/24/ubuntu.sources        (zoom-debian-virtual)
#   - arm64 -> ubuntu/24/arm64/ubuntu.sources  (public-debian-ubuntu-arm)
#
# `docker buildx build --load` only supports a SINGLE platform (a multi-arch manifest
# cannot be loaded into the local image store). So this script builds ONE arch per run and
# loads it. To get both arches, run twice with ARCH=amd64 and ARCH=arm64.
#
# Usage (invoked by Zoom CI; MODULE_NAME comes from ci/config.txt's ZOOMCI_HOOK_BUILD_ENV):
#   ARCH=amd64 MODULE_NAME=nemo-rl bash ci/build.sh
#   ARCH=arm64 MODULE_NAME=nemo-rl bash ci/build.sh
#
# Env vars:
#   ARCH         target arch: "amd64" (default) or "arm64".
#   MODULE_NAME  Zoom CI module name (default: nemo-rl). Final tags are
#                zoomai/mlp/${MODULE_NAME} and zoomai/mlp/${MODULE_NAME}-${ARCH}.

set -euo pipefail

# ---- config (override via env) ----
ARCH="${ARCH:-amd64}"
MODULE_NAME="${MODULE_NAME:-nemo-rl}"
DOCKERFILE="ci/Dockerfile"
# -----------------------------------

case "${ARCH}" in
    amd64 | arm64) ;;
    *)
        echo "ERROR: unsupported ARCH='${ARCH}' (expected 'amd64' or 'arm64')" >&2
        exit 1
        ;;
esac

PLATFORM="linux/${ARCH}"
# Tag the Zoom CI publish stage expects (no arch suffix), plus an arch-suffixed tag.
PUBLISH_TAG="zoomai/mlp/${MODULE_NAME}"
IMAGE_TAG="${PUBLISH_TAG}-${ARCH}"

# Resolve repo root = parent dir of this script's dir (ci/ -> repo root).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)"
cd "${REPO_ROOT}"

echo "==> Repo root:    ${REPO_ROOT}"
echo "==> Module name:  ${MODULE_NAME}"
echo "==> Arch:         ${ARCH}"
echo "==> Platform:     ${PLATFORM}"
echo "==> Arch tag:     ${IMAGE_TAG}"
echo "==> Publish tag:  ${PUBLISH_TAG}"
echo "==> Dockerfile:   ${DOCKERFILE}"

# Regenerate the container fingerprint from the CURRENT code so runtime no longer
# reports Code/Container mismatch.
echo "==> Generating fingerprint -> ci/nemo_rl_container_fingerprint"
python3 tools/generate_fingerprint.py > ci/nemo_rl_container_fingerprint
cat ci/nemo_rl_container_fingerprint

# ---- Single-arch build via buildx, loaded into the local image store ----
# Mirror ml-runtime-env/ci/build.sh: use Zoom's internal buildkit image (docker.io is
# unreachable from the CI network, so `tonistiigi/binfmt` cannot be pulled). The
# docker-container driver + this buildkit image support cross-arch (amd64/arm64) builds.
export DOCKER_CLI_EXPERIMENTAL=enabled

docker pull artifacts.corp.zoom.us/zoom-docker-virtual/moby/buildkit:buildx-stable-1
docker tag artifacts.corp.zoom.us/zoom-docker-virtual/moby/buildkit:buildx-stable-1 moby/buildkit:buildx-stable-1

BUILDER="multiple-arch-simulate"
if ! docker buildx inspect "${BUILDER}" >/dev/null 2>&1; then
    echo "==> Creating buildx builder ${BUILDER}"
    docker buildx create --use --name "${BUILDER}" \
        --driver-opt image=moby/buildkit:buildx-stable-1
fi
docker buildx use "${BUILDER}"

# --load loads the built single-platform image into the local docker image store
# (no registry push needed). We tag both the publish name and the arch-suffixed name so
# the Zoom CI publish stage can find zoomai/mlp/${MODULE_NAME}.
echo "==> Building (${PLATFORM}) and loading into local docker: ${IMAGE_TAG} (+ ${PUBLISH_TAG})"
DOCKER_BUILDKIT=1 docker buildx build \
    --platform "${PLATFORM}" \
    -f "${DOCKERFILE}" \
    -t "${IMAGE_TAG}" \
    -t "${PUBLISH_TAG}" \
    --load \
    "${REPO_ROOT}"
echo "==> Build + load complete: ${IMAGE_TAG} (+ ${PUBLISH_TAG})"
docker images "zoomai/mlp/${MODULE_NAME}" || true

echo "==> Done. Runtime reminder: set NEMO_RL_HOME=/opt/nemo-rl (and NEMO_RL_PROJECT_ROOT/UV_PROJECT_DIR),"
echo "    run from /opt/nemo-rl, and do NOT mount the network FS over /opt/nemo-rl."
