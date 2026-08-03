# Speed up Ray worker startup: bake NeMo-RL code onto the image's local disk

## Background: why Ray workers are slow

Confirmed from the training log:

- **Not a missing dependency, and not a venv rebuild.** All workers reuse the image's
  prebuilt
  `/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker`;
  there is no `Creating new venv`.
- The real bottleneck is the **cold import over the network FS**: the training script
  points `NEMO_RL_HOME` at the network FS `/code/model_dev/RL`, so every worker doing
  `import megatron.core.*` has to `stat/open` thousands of small `.py` files on the
  network FS. From the log:

  ```
  Initializing lm_policy workers:   6%| | 1/16 [05:28<1:22:08, 328.59s/worker]
  ...
  ✓ 16 workers initialized in 446.99s
  ```

  **The first worker alone took 5 min 28 s** (later workers drop to 8–28 s once the
  network/OS cache is warm). All megatron import-warning paths are
  `/code/model_dev/RL/3rdparty/...`, i.e. the network FS, which confirms the source.

## Key mechanism (why this fix works)

- Whether a venv is rebuilt is decided ONLY by `NRL_FORCE_REBUILD_VENVS` and whether
  `$NEMO_RL_VENV_DIR/<fqn>/bin/python` exists (see the early return in
  `nemo_rl/utils/venvs.py:119`) — it is **unrelated to the fingerprint**. The fingerprint
  check (`_check_container_fingerprint()` in `nemo_rl/__init__.py:202`) only prints a warning.
- In the image, `nemo_rl / megatron.core / megatron.bridge / nemo_automodel / nemo_gym`
  are all **editable installs**, and their finder `MAPPING` is **hardcoded to
  `/opt/nemo-rl/...`**. So as long as the code under `/opt/nemo-rl` is the code you want to
  run, imports read from local disk.

## Solution (Plan 2): bake the current repo into `/opt/nemo-rl`

The following files have been generated under `ci/`:

| File | Purpose |
|---|---|
| `ci/Dockerfile` | Based on `nvcr.io/nvidia/nemo-rl:v0.6.0`; installs `sudo`+`python3-dev`, `rm -rf /opt/nemo-rl` then `COPY . /opt/nemo-rl`, writes the prebaked fingerprint, and **fully keeps your original Dockerfile's fine-grained permission grants** (placed after COPY so the new code dir is also granted; non-root can run) |
| `ci/build.sh` | One-shot build: submodule check → fingerprint regen → BuildKit build → optional push |
| `ci/nemo_rl_container_fingerprint` | Fingerprint precomputed on the build host (includes the 4 submodule SHAs); matches the current code so runtime no longer reports mismatch |
| `ci/Dockerfile.dockerignore` / `ci/.dockerignore` | Exclude `.git` (~470M) etc. to avoid copying into the image |

---

## Steps

### Step 1+2: One-shot build (recommended: use ci/build.sh)

`ci/build.sh` wraps "check submodules → regen fingerprint → BuildKit build → (optional) push":

```bash
# Default tag is date-stamped (YYYYMMDD), e.g. nemo-rl:v0.6.0-20260623
bash ci/build.sh

# Specify image name/tag:
IMAGE=registry.zoomdev.us/languagetech/nemo-rl:v0.6.0-20260623 bash ci/build.sh

# Build and push:
PUSH=1 IMAGE=registry.zoomdev.us/languagetech/nemo-rl:v0.6.0-20260623 bash ci/build.sh

# Build and push with auto-login (avoids 'unauthorized' from stale docker creds):
PUSH=1 REGISTRY_USER=admin REGISTRY_PASS='<password>' \
  IMAGE=registry.zoomdev.us/languagetech/nemo-rl:v0.6.0-20260623 bash ci/build.sh
```

> Push auth: the script derives the registry host from `IMAGE` and runs `docker login`
> before pushing — using `REGISTRY_USER`/`REGISTRY_PASS` if provided, otherwise refreshing
> your stored credentials. If a push still fails with `unauthorized` (stale token), it
> re-logins and retries once. You can also just run `docker login registry.zoomdev.us`
> yourself beforehand.

#### Multi-platform build (amd64 + arm64)

The base image `nvcr.io/nvidia/nemo-rl:v0.6.0` is a multi-arch manifest list
(linux/amd64 + linux/arm64), so you can build a multi-arch image too. Set `PLATFORMS`
and `PUSH=1` (a manifest list cannot be loaded into the local docker store, so it must be
pushed directly):

```bash
PUSH=1 PLATFORMS="linux/amd64,linux/arm64" \
  REGISTRY_USER=admin REGISTRY_PASS='<password>' \
  IMAGE=registry.zoomdev.us/languagetech/nemo-rl:v0.6.0-20260624 bash ci/build.sh
```

What the multi-platform path does (in `ci/build.sh`):
1. Registers QEMU emulators via `docker run --privileged tonistiigi/binfmt --install all`
   so arm64 `RUN` steps can run on an amd64 host (emulated — the arm64 `apt-get`/`chmod`
   steps are noticeably slower).
2. Creates/uses a `docker-container` buildx builder (`nemo-rl-multiarch-builder`).
3. Runs `docker buildx build --platform linux/amd64,linux/arm64 ... --push`, producing a
   single manifest-list tag that serves both architectures.

Notes:
- `PLATFORMS` unset (or `linux/amd64`) → plain single-arch `docker build` (loads locally).
- Multi-platform **requires** `PUSH=1`.
- If you have a native arm64 node (e.g. a GB200), a cross-node buildx builder is faster
  than QEMU; but QEMU is the simplest single-host option.



What the script does:
1. `git submodule update --init` and verify the 4 submodule sources are present
   (prerequisite for local-disk imports);
2. `python3 tools/generate_fingerprint.py > ci/nemo_rl_container_fingerprint`;
3. `DOCKER_BUILDKIT=1 docker build -f ci/Dockerfile -t $IMAGE .` (BuildKit uses
   `ci/Dockerfile.dockerignore` to exclude `.git`);
4. `docker push $IMAGE` when `PUSH=1`.

> Equivalent manual commands:
> ```bash
> cd /share5/users/chengyi.wu/train/RL
> git submodule update --init
> python3 tools/generate_fingerprint.py > ci/nemo_rl_container_fingerprint
> DOCKER_BUILDKIT=1 docker build -f ci/Dockerfile -t <registry>/nemo-rl:v0.6.0-<date> .
> ```

### Step 3: Update the training script so the code root points to the image's local disk

Change these in your training script from the network FS to `/opt/nemo-rl`:

```bash
export NEMO_RL_HOME=/opt/nemo-rl
export NEMO_RL_PROJECT_ROOT=/opt/nemo-rl
export UV_PROJECT_DIR=/opt/nemo-rl

cd /opt/nemo-rl && \
NRL_FORCE_REBUILD_VENVS=false NCCL_NVLS_ENABLE=0 \
  uv run --no-sync /opt/nemo-rl/examples/run_dpo.py \
  --config ${TRAINING_CODE_REPO_PATH}/configs/<your-config>.yaml \
  ...other overrides unchanged...
```

- Your configs (`TRAINING_CODE_REPO_PATH=/code/nemo_rl_training`) **can stay on the
  network FS**, since reading a few yamls does not affect import speed.
- **Important: do NOT mount the network FS over `/opt/nemo-rl` at runtime**, otherwise
  imports fall back to the network FS again.

### About permissions (kept from your original Dockerfile)

Step 4 of `ci/Dockerfile` **fully preserves your original Dockerfile's fine-grained chmod
grants** (dirs 777; bin/venv/ray/python/pip executables 755; site-packages writable 666;
others 644), and runs **after `COPY . /opt/nemo-rl`** — so the newly copied code dir is
also granted, and non-root users can run normally. It does not weaken the existing grants
of `/opt/ray_venvs` or `/opt/nemo_rl_venv` in the image.

### Step 4: Verify it works

Run a training job once and confirm in the log:

- Worker import paths become `/opt/nemo-rl/3rdparty/...` (local disk), not `/code/.../3rdparty/...`;
- The first worker time in `Initializing lm_policy workers` drops from ~5.5 min to seconds;
- No more `Container/Code Version Mismatch` warning;
- No more `make: python3-config: No such file or directory`.

---

## FAQ

- **What if I change NeMo-RL code later?** Re-run Step 1 (regen fingerprint) + Step 2
  (rebuild) — i.e. just re-run `bash ci/build.sh`. If you only changed `.py` logic without
  touching `pyproject.toml`/`uv.lock`/submodules, the fingerprint is unchanged and you can
  just rebuild (`COPY . /opt/nemo-rl` picks up the new code).
- **Will dependencies conflict with the prebuilt venvs?** No. v0.6.0 and this repo
  (fbb86eeba) resolve to the same package set on linux x86_64 (only soundfile moved from a
  transitive to a direct dependency + platform marker tweaks), so the `/opt/ray_venvs/*`
  prebuilt venvs can be reused as-is.
- **What if I truly need new dependencies?** Then you'd need to rebuild the venv inside the
  image with `uv sync --extra mcore` (or the relevant extra) and update the fingerprint
  accordingly; not needed for the current scenario.
