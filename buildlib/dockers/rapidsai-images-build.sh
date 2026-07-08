#!/bin/bash -eE
#
# Build + push the UCXX rapidsai CI wrapper images, multi-arch (amd64 + arm64),
# natively via a 2-node buildx builder + `docker buildx bake` -- one command
# builds both arches on their native node and writes the manifest.
#
# Run on either builder host; it takes the local arch as-is and appends the
# opposite arch over an ssh docker context. Set the ssh endpoint of the OTHER
# builder and the harbor push credentials in the environment:
#
#   RAPIDS_VER=<ver> \
#   ARM_NODE=ssh://<user>@<arm-builder>   # required when running on an x86 host
#   X86_NODE=ssh://<user>@<x86-builder>   # required when running on an arm host
#   HARBOR_UCX_USER=... HARBOR_UCX_PASSWORD=... \
#   ./rapidsai-images-build.sh
#
# The running host must key-auth (non-interactive) ssh to the other node; the
# remote buildkit runs over that ssh context. Do not leave credentials at rest
# on the builder -- pass them via the environment; the script logs out on exit.
#
# Env:
#   RAPIDS_VER   required, RAPIDS CalVer, e.g. 26.08
#   ARM_NODE     ssh endpoint of the arm64 builder (required on an x86 host)
#   X86_NODE     ssh endpoint of the x86_64 builder (required on an arm host)
#   REGISTRY     push registry (default: harbor.mellanox.com)
#   BUILDER      buildx builder name (default: ucx-rapidsai)
#   HARBOR_UCX_USER / HARBOR_UCX_PASSWORD   harbor push credentials (required)

basedir=$(cd "$(dirname "$0")" && pwd)
: "${RAPIDS_VER:?set RAPIDS_VER, e.g. 26.08}"
: "${HARBOR_UCX_USER:?set HARBOR_UCX_USER}" "${HARBOR_UCX_PASSWORD:?set HARBOR_UCX_PASSWORD}"
BUILDER=${BUILDER:-ucx-rapidsai}
REMOTE_CTX=${REMOTE_CTX:-ucx-remote-buildnode}
REGISTRY=${REGISTRY:-harbor.mellanox.com}

# Run from EITHER builder: the local arch is native, the opposite arch is
# appended over ssh. Only the opposite-arch endpoint is needed.
case "$(uname -m)" in
  x86_64)  LOCAL_PLAT=linux/amd64; REMOTE_PLAT=linux/arm64
           REMOTE_NODE=${ARM_NODE:?set ARM_NODE=ssh://<user>@<arm-builder>} ;;
  aarch64) LOCAL_PLAT=linux/arm64; REMOTE_PLAT=linux/amd64
           REMOTE_NODE=${X86_NODE:?set X86_NODE=ssh://<user>@<x86-builder>} ;;
  *) echo "ERROR: unexpected host arch $(uname -m)" >&2; exit 1 ;;
esac

# 1. preflight - every BASE_IMAGE in the compose must exist. The compose is
# the single source for the base tags; this parses them instead of keeping a
# second copy that drifts when RAPIDS moves a cuda/py version.
while read -r t; do
  t="${t//\$\{RAPIDS_VER\}/$RAPIDS_VER}"
  echo ">> preflight: $t"
  docker manifest inspect "$t" >/dev/null \
    || { echo "ERROR: base image not found: $t (did RAPIDS move the cuda/py tag?)" >&2; exit 1; }
done < <(sed -n 's/^[[:space:]]*BASE_IMAGE:[[:space:]]*//p' "$basedir/docker-compose-rapidsai.yml")

# 2. native 2-node buildx builder (idempotent): local node = host arch, remote
# node = opposite arch over ssh. --platform pins each node to ONE arch so buildx
# can't route the foreign arch to the local node's QEMU (crawls on multi-GB images).
# The buildkit image is pinned: newer runc in the floating buildx-stable tag
# fails to start containers on the builders' 5.4-era kernels
# ("can't mask dir /proc/acpi ... invalid argument").
BUILDKIT_IMAGE=${BUILDKIT_IMAGE:-moby/buildkit:v0.23.2}
if ! docker buildx inspect "$BUILDER" >/dev/null 2>&1; then
  docker context inspect "$REMOTE_CTX" >/dev/null 2>&1 \
    || docker context create "$REMOTE_CTX" --docker "host=$REMOTE_NODE"
  docker buildx create --name "$BUILDER" --driver docker-container \
    --driver-opt "image=$BUILDKIT_IMAGE" \
    --platform "$LOCAL_PLAT" --bootstrap                                                # local (host arch)
  docker buildx create --name "$BUILDER" --append --node "${BUILDER}-remote" \
    --driver-opt "image=$BUILDKIT_IMAGE" \
    --platform "$REMOTE_PLAT" "$REMOTE_CTX"                                             # remote (opposite arch)
fi
docker buildx inspect "$BUILDER" --bootstrap >/dev/null

# 3. login -> bake (builds both arches on their native node, pushes, writes manifest) -> logout.
# Platforms are forced on the CLI: bake does not honor the compose
# `build.platforms` key and would otherwise build only the local arch.
printf '%s' "$HARBOR_UCX_PASSWORD" | docker login "$REGISTRY" -u "$HARBOR_UCX_USER" --password-stdin
trap 'docker logout "$REGISTRY" >/dev/null 2>&1 || true' EXIT
RAPIDS_VER="$RAPIDS_VER" docker buildx bake \
  --builder "$BUILDER" -f "$basedir/docker-compose-rapidsai.yml" \
  --set '*.platform=linux/amd64,linux/arm64' --push

# 4. verify the 3 manifests are multi-arch
echo "=== pushed manifests ==="
for img in rapidsai-ci-conda:${RAPIDS_VER}-azp-1 \
           rapidsai-ci-wheel:${RAPIDS_VER}-cuda12-azp-1 \
           rapidsai-ci-wheel:${RAPIDS_VER}-cuda13-azp-1; do
  printf '%-42s ' "$img"
  docker manifest inspect "$REGISTRY/ucx/$img" 2>/dev/null \
    | grep -o '"architecture": "[a-z0-9]*"' | tr '\n' ' '
  echo
done
echo "DONE: rapidsai-ci-{conda,wheel cuda12/13}:${RAPIDS_VER}-azp-1 (amd64+arm64)"
