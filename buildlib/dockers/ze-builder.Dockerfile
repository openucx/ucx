# UCX Level Zero (ZE) builder image.
#
# Local build:
#   docker build -t ze-builder:local \
#       -f buildlib/dockers/ze-builder.Dockerfile buildlib/dockers
#
# CI publish: this image is pulled by the ubuntu2404_ze container in
# buildlib/pr/main.yml. Build and push it to the internal registry under a
# unique, content-reflecting tag (per buildlib/azure-pipelines.md), e.g.
#   rdmz-harbor.rdmz.labs.mlnx/hpcx/x86_64/ubuntu24.04/ze-builder:oneapi-2026.0.0-l0-1.27.0
# and keep main.yml's image ref in sync with the published tag.
#
# The Intel oneAPI toolkit base already ships the Level Zero loader, headers and
# pkg-config, which is what UCX needs to configure --with-ze.
FROM intel/oneapi-toolkit:2026.0.0-devel-ubuntu24.04

# The null driver is NOT shipped by any release/runtime package — it only builds
# with BUILD_L0_LOADER_TESTS=1 + INSTALL_NULL_DRIVER=1. We build it from the
# level-zero source tag that MATCHES the loader already present in this image,
# then install ONLY libze_null.so.* so the image's loader stays authoritative.
# The match matters: the loader dlopens "libze_null.so.<loader_version>"
# (MAKE_LIBRARY_NAME), so a version-mismatched null driver would not load.
#
# Tag of the level-zero source to build the null driver from. Must match the
# loader version the base image ships (verified: intel/oneapi-toolkit:2026.0.0
# ships loader 1.27.0). If set empty (ARG L0_TAG=), the build auto-detects the
# tag from the image's libze_loader.so.1 soname instead of using this default.
# Override: docker build --build-arg L0_TAG=v1.30.0
ARG L0_TAG=v1.27.0

# Build deps for level-zero (cmake/ninja/git) + UCX (autotools, IB headers).
# Base image already provides a C/C++ toolchain.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        cmake \
        debhelper \
        doxygen \
        dpkg-dev \
        fakeroot \
        git \
        libibverbs-dev \
        librdmacm-dev \
        libtool \
        m4 \
        make \
        ninja-build \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Build the in-tree null driver at the image loader's version and install ONLY it.
RUN set -eux; \
    # 1. Locate the versioned loader the image ships. Prefer the ldconfig cache,
    #    then fall back to a filesystem search (oneAPI may install outside the
    #    default linker path).
    loader_real="$(ldconfig -p | awk '/libze_loader\.so\.1 /{print $NF; exit}')"; \
    loader_real="$([ -n "${loader_real}" ] && readlink -f "${loader_real}" || true)"; \
    case "${loader_real}" in \
        *libze_loader.so.[0-9]*.[0-9]*.[0-9]*) ;; \
        *) loader_real="$(find / -xdev -name 'libze_loader.so.[0-9]*.[0-9]*.[0-9]*' 2>/dev/null | head -n1)" ;; \
    esac; \
    test -n "${loader_real}" || { echo "ERROR: image has no versioned libze_loader.so"; exit 1; }; \
    img_ver="$(printf '%s' "${loader_real}" | sed -nE 's/.*libze_loader\.so\.([0-9]+\.[0-9]+\.[0-9]+)$/\1/p')"; \
    echo "Image loader: ${loader_real} (version ${img_ver:-unknown})"; \
    # 2. Pick the source tag: explicit ARG wins, else the detected vX.Y.Z.
    tag="${L0_TAG:-${img_ver:+v${img_ver}}}"; \
    test -n "${tag}" || { echo "ERROR: could not detect loader version; pass --build-arg L0_TAG=vX.Y.Z"; exit 1; }; \
    echo "Building null driver from level-zero ${tag}"; \
    # 3. Clone that exact tag (fail loudly if the tag does not exist upstream).
    git clone --depth 1 --branch "${tag}" \
        https://github.com/oneapi-src/level-zero.git /tmp/level-zero; \
    # 4. Configure with the test/null-driver targets enabled.
    cmake -S /tmp/level-zero -B /tmp/level-zero/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_L0_LOADER_TESTS=1 \
        -DINSTALL_NULL_DRIVER=1; \
    # 5. Build ONLY the null driver (we do NOT run `install` — that would
    #    overwrite the image loader). zello_world is built best-effort below.
    cmake --build /tmp/level-zero/build --target ze_null; \
    # 6. Install just the null driver shared object + symlinks onto the linker path.
    cp -P /tmp/level-zero/build/lib/libze_null.so* /usr/local/lib/; \
    ldconfig; \
    # 7. Hard build-time gate: the versioned null-driver soname must be resolvable.
    ldconfig -p | grep -F 'libze_null.so.1'; \
    # 8. Best-effort smoke: build+run zello_world IF the sample target is present
    #    in this tag. The authoritative load gate is ucx_info in builds.sh; here
    #    we only fail the image build when a sample that DID build then misbehaves.
    if cmake --build /tmp/level-zero/build --target zello_world 2>/dev/null \
        && [ -x /tmp/level-zero/build/bin/zello_world ]; then \
        ZE_ENABLE_NULL_DRIVER=1 ZE_ENABLE_LOADER_DEBUG_TRACE=1 \
            /tmp/level-zero/build/bin/zello_world; \
    else \
        echo "zello_world sample not available for ${tag}; skipping in-image smoke"; \
    fi; \
    # 9. Clean up source tree (driver already installed).
    rm -rf /tmp/level-zero
