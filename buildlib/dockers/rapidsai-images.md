# UCXX CI: RAPIDS wrapper images

The UCXX jobs in the openucx/ucx Azure pipelines run inside thin wrapper images
built on top of upstream RAPIDS CI images. This directory holds the Dockerfiles,
a compose file, and a build script to (re)build and push them.

| Wrapper image (`ucx/`)                     | RAPIDS base                                            | Used for                   |
|--------------------------------------------|--------------------------------------------------------|----------------------------|
| `rapidsai-ci-conda:<VER>-azp-1`            | `rapidsai/ci-conda:<VER>-latest`                       | conda build + tests        |
| `rapidsai-ci-wheel:<VER>-cuda13-azp-1`     | `rapidsai/ci-wheel:<VER>-cuda<CU13>-rockylinux8-py3.11`| libucxx/ucxx wheels (CUDA 13) |
| `rapidsai-ci-wheel:<VER>-cuda12-azp-1`     | `rapidsai/ci-wheel:<VER>-cuda<CU12>-rockylinux8-py3.11`| libucxx/ucxx wheels (CUDA 12) |

`<VER>` is the RAPIDS CalVer, e.g. `26.08`. The wrappers only add a `chmod` so the
non-root UID Azure runs steps as can write (`/opt/conda`, `/pyenv`) and `gdb` for
stack capture. Each image is a multi-arch manifest (`linux/amd64` + `linux/arm64`);
Azure picks the right arch per agent automatically.

## When to rebuild

RAPIDS ships ~every 2 months (CalVer `YY.MM`). We do **not** bump on every release:
the ucxx repo is pinned to a specific `main` commit, so we rebuild these images only
when we intentionally advance that pin to a commit whose base images/scripts require
a newer RAPIDS version. The CI images, the ucxx ref, and the in-repo CI scripts move
together.

## Prerequisites

- Two builder hosts, one `x86_64` and one `aarch64`, each with Docker + `buildx`.
- The host you run on must key-auth (non-interactive) ssh to the other host - the
  remote buildkit runs over that ssh context. One direction suffices: run the
  script from whichever host holds the key to the other.
- Harbor `ucx/` push credentials (`HARBOR_UCX_USER` / `HARBOR_UCX_PASSWORD`).

(The specific builder hosts and credential source are environment-specific and are
not committed here - supply them via the environment, see below.)

## Procedure

### 1. Confirm the new RAPIDS base tags exist

The build script preflights this; to check by hand (no login needed):

```sh
VER=26.08
docker manifest inspect rapidsai/ci-conda:${VER}-latest
docker manifest inspect rapidsai/ci-wheel:${VER}-cuda<CU13>-rockylinux8-py3.11
docker manifest inspect rapidsai/ci-wheel:${VER}-cuda<CU12>-rockylinux8-py3.11
```

The exact CUDA (`<CU12>`/`<CU13>`) and Python versions live in ONE place:
the `BASE_IMAGE` args of `docker-compose-rapidsai.yml` (the build script derives
its preflight from them). If RAPIDS moved a CUDA or Python version, update the
compose **and** the matching `rapids_cuda_version` / `rapids_py_version` in
`pr/main.yml` + `azure-pipelines-release.yml`.

### 2. Build + push all three images, multi-arch

One command, run from either builder (the script detects the host arch and appends
the opposite arch over ssh). Set the opposite-arch node endpoint and harbor creds in
the environment:

```sh
RAPIDS_VER=26.08 \
ARM_NODE=ssh://<user>@<arm-builder> \   # when running on an x86 host
HARBOR_UCX_USER=... HARBOR_UCX_PASSWORD=... \
./rapidsai-images-build.sh
```

The script preflights the base tags, creates a 2-node buildx builder (each node
pinned to its native arch so buildx never falls back to slow QEMU emulation),
logs in to harbor, runs `docker buildx bake --push` (builds both arches on their
native node and writes the manifest in one shot), logs out, and verifies each
pushed tag is multi-arch.

Credential hygiene: pass creds via the environment, never leave them at rest on a
builder. The script logs out on exit.

### 3. Point the pipelines at the new images

Bump the old `<VER>` to the new one in the only places that reference it:

- `buildlib/dockers/rapidsai-ci-conda.Dockerfile` (`BASE_IMAGE`)
- `buildlib/dockers/rapidsai-ci-wheel.Dockerfile` (`BASE_IMAGE`)
- `buildlib/pr/main.yml` (image tags)
- `buildlib/azure-pipelines-release.yml` (image tags)

```sh
grep -rn <VER> buildlib/dockers buildlib/pr/main.yml buildlib/azure-pipelines-release.yml
python3 -c "import yaml; yaml.safe_load(open('buildlib/pr/main.yml'))"
python3 -c "import yaml; yaml.safe_load(open('buildlib/azure-pipelines-release.yml'))"
```

Commit on the PR-pipeline branch, rebase the release-pipeline branch onto it (so it
inherits the Dockerfile + `main.yml` bump), bump the release.yml tags there, push,
and re-run CI.

## Why multi-arch + buildx

The legacy UCX release images (`docker-compose-<arch>.yml` + `push-release-images.sh`)
build each arch separately and push to per-arch paths (`ucx/x86_64/...`,
`ucx/aarch64/...`), predating good multi-arch tooling. The RAPIDS wrappers use the
modern path: one compose file with `build.platforms`, built via `docker buildx bake`
against a 2-node native builder, producing a single multi-arch manifest per image -
fewer pipeline container resources (one tag, arch auto-resolved) and one command
instead of a per-arch loop plus manual `imagetools create`.

## Troubleshooting

- **only one architecture in the pushed manifest** - `docker buildx bake` does not
  honor the compose `build.platforms` key; the script forces both platforms with
  `--set '*.platform=...'`. If you bake by hand, pass it too.
- **`runc run failed ... can't mask dir /proc/acpi`** - the floating
  `moby/buildkit:buildx-stable-1` image ships a runc that fails on the builders'
  older kernels; the script pins the buildkit image (`BUILDKIT_IMAGE`).
- **`no space left on device` under `/var/lib/buildkit`** - the buildx builder's
  own container volume filled up. `docker buildx rm <builder>` and rerun (the
  script recreates it).
- **arm64 build extremely slow** - it landed on the x86 node's QEMU emulation instead
  of the native arm node. The `--platform` pins in the script prevent this; if you
  build by hand, pin each node to one arch.
- **`docker context create` / ssh fails** - the host you run on can't key-auth to the
  other node. Fix ssh keys first; the remote buildkit runs over that ssh context.
- **manifest shows only one arch after a push** - a node's push didn't complete (seen
  when a backgrounded ssh build was killed early). Re-push the missing arch image and
  re-create the manifest (`docker buildx imagetools create`), or just re-run the bake.
- **`docker login` 401 on `ucx/` repos** - the cached login is for a different harbor
  project. Re-login with `HARBOR_UCX_USER`.
