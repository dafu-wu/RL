# Based on the official NeMo-RL v0.6.0 image, bake the CURRENT RL repo source into the
# in-image local disk /opt/nemo-rl, so that Ray worker imports
# (nemo_rl / megatron.core / megatron.bridge / nemo_automodel / nemo_gym are all editable
# installs whose finder MAPPING is hardcoded to /opt/nemo-rl/...) read from local disk,
# avoiding the cold import (~5.5 min for the first worker) caused by pointing NEMO_RL_HOME
# at the network FS (/code/...) at runtime.
#
# Dependency note: v0.6.0 and this repo (fbb86eeba) resolve to the same package set on
# linux x86_64 (only soundfile moved from a transitive to a direct dependency + platform
# marker tweaks), so the prebuilt /opt/ray_venvs/* venvs can be reused as-is, no rebuild.
#
# Build (run from the RL repo root with submodules already initialized; the context MUST
# contain the 3rdparty/*/ source):
#   DOCKER_BUILDKIT=1 docker build -f ci/Dockerfile \
#       -t <registry>/nemo-rl:v0.6.0-chengyi .
#   # Note: BuildKit prefers the Dockerfile-adjacent ci/Dockerfile.dockerignore, which
#   #       excludes .git (~470M) from being copied into the context/image.
#   #       (The root .dockerignore does NOT exclude .git, so rely on the per-Dockerfile one.)
#
# Runtime notes (in your training script):
#   export NEMO_RL_HOME=/opt/nemo-rl          # do NOT point to network FS; do NOT mount over it
#   export NEMO_RL_PROJECT_ROOT=/opt/nemo-rl
#   export UV_PROJECT_DIR=/opt/nemo-rl
#   cd /opt/nemo-rl && uv run --no-sync /opt/nemo-rl/examples/run_dpo.py --config <your network yaml> ...
#   # Your configs (TRAINING_CODE_REPO_PATH) can stay on the network FS; reading a few yamls
#   # does not affect import speed.

# Use Zoom's internal mirror of the NeMo-RL v0.6.0 base image (nvcr.io is
# unreachable from the CI network).
FROM artifacts.corp.zoom.us/zoom-docker-virtual/nvidia/nemo-rl:v0.7.0

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive
# BuildKit auto-populates TARGETARCH/TARGETPLATFORM with the TARGET platform ("amd64" or
# "arm64") for each platform of a multi-arch build -- but ONLY if the ARG is (re)declared
# in the build stage that uses it, WITHOUT a hardcoded default (a default like "=amd64"
# shadows the auto value, which is why the arm64 layer previously resolved to amd64 and
# pulled amd64-only packages). We declare TARGETPLATFORM too as a robust fallback.
ARG TARGETARCH
ARG TARGETPLATFORM


# Point apt at Zoom's internal mirror (Ubuntu 24/Noble, deb822 format), since the
# base image's default Ubuntu/NVIDIA apt sources are unreachable from the CI network.
# The internal mirror stores the two arches at DIFFERENT paths / repos:
#   - amd64: ubuntu/24/ubuntu.sources        -> zoom-debian-virtual
#            (Signed-By zoomci.gpg, which is NOT in the base image, so we ADD it here).
#            NOTE: there is NO ubuntu/24/amd64/ubuntu.sources (that dir only holds the
#            nvidia/cuda keyring+list), so amd64 must use the top-level ubuntu.sources.
#   - arm64: ubuntu/24/arm64/ubuntu.sources  -> public-debian-ubuntu-arm
#            (Signed-By ubuntu-archive-keyring.gpg, already shipped in the base image).
# We copy the zoomci keyring unconditionally (only referenced by the amd64 source list)
# and select the per-arch source list based on the TARGET architecture.
RUN echo "Acquire { https::Verify-Peer false }" >> /etc/apt/apt.conf.d/70debconf && \
    rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources
ADD https://artifacts.corp.zoom.us:443/artifactory/zoom-generic-thirdparty/jfrog/repoconfig/ubuntu/24/zoomci.gpg /usr/share/keyrings/zoomci.gpg
# Resolve the target arch: prefer TARGETARCH, fall back to the last path segment of
# TARGETPLATFORM (e.g. "linux/arm64" -> "arm64"). arm64 -> "ubuntu/24/arm64", everything
# else (amd64) -> top-level "ubuntu/24".
RUN ARCH="${TARGETARCH:-${TARGETPLATFORM##*/}}"; \
    echo "Resolved target arch: '${ARCH}' (TARGETARCH='${TARGETARCH}' TARGETPLATFORM='${TARGETPLATFORM}')"; \
    if [ "${ARCH}" = "arm64" ]; then ARM_SUBDIR="arm64/"; else ARM_SUBDIR=""; fi; \
    curl -fsSLk \
      "https://artifacts.corp.zoom.us:443/artifactory/zoom-generic-thirdparty/jfrog/repoconfig/ubuntu/24/${ARM_SUBDIR}ubuntu.sources" \
      -o /etc/apt/sources.list.d/ubuntu.sources

RUN chmod 0644 /usr/share/keyrings/zoomci.gpg && \
    apt-get update && \
    apt-get install -y ca-certificates && \
    sed -i '$d' /etc/apt/apt.conf.d/70debconf





# 1) sudo (for non-root usage) + python3-dev (removes the per-worker
#    "make: python3-config: No such file or directory" noise)
RUN apt-get update && \
    apt-get install -y --no-install-recommends sudo python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# 2) Overlay the current repo source onto the in-image /opt/nemo-rl (local disk).
#    We do NOT `rm -rf /opt/nemo-rl` and do NOT copy 3rdparty/*: the base image already
#    ships the v0.6.0 submodule source as editable installs, and 3rdparty/ is excluded via
#    ci/Dockerfile.dockerignore. So this COPY only overwrites the changed main-repo files
#    (nemo_rl/, examples/, etc.) on top of the existing tree, leaving submodules untouched.
#    (.git is also excluded via ci/Dockerfile.dockerignore to keep the image small.)
COPY . /opt/nemo-rl


# 3) Write the fingerprint PRECOMPUTED ON THE BUILD HOST (includes submodule SHAs) so that
#    nemo_rl/__init__.py's _check_container_fingerprint() no longer reports Code/Container
#    mismatch at runtime.
#    (Generate on the build host with:
#       python3 tools/generate_fingerprint.py > ci/nemo_rl_container_fingerprint )
COPY ci/nemo_rl_container_fingerprint /opt/nemo_rl_container_fingerprint

# 4) Adjust permissions so non-root users can also use it.
#
#    NOTE on layer size: granting write under the venv site-packages necessarily rewrites
#    a lot of base-image inode metadata, which Docker records as a new layer. We keep it as
#    small as reasonably possible by:
#      (a) granting only the freshly-copied code dir /opt/nemo-rl, and
#      (b) recursively making the venv site-packages trees writable (a+rwX) so non-root can
#          `pip install` new packages. This (b) layer is multi-GB by nature (chosen
#          intentionally); if push is too slow, prefer granting at runtime instead.
RUN set -eux; \
    chmod 755 /root; \
    chmod -R 755 /root/.local 2>/dev/null || true; \
    # (a) grant the new code dir (small)
    chmod -R a+rX /opt/nemo-rl; \
    find /opt/nemo-rl -type d -exec chmod 777 {} +; \
    find /opt/nemo-rl -type f -name "*.sh" -exec chmod 755 {} + 2>/dev/null || true; \
    # (b) make venv site-packages writable so non-root can pip install new packages
    #     (POSIX sh compatible: chmod -R directly on every site-packages dir found)
    find /opt/nemo_rl_venv /opt/ray_venvs -maxdepth 4 -type d -name site-packages \
        -exec chmod -R a+rwX {} + 2>/dev/null || true


WORKDIR /opt/nemo-rl
CMD ["/bin/bash"]


