# UCX Level Zero (ZE) CI builder image.
#
# Derived from intel/deep-learning-essentials so we get the Level Zero
# loader, headers and pkg-config out of the box, plus the autotools and
# doxygen that the UCX build requires.  This mirrors what the CUDA and
# ROCm CI lanes do: the public NVIDIA / AMD images are wrapped in a
# small builder image that has all the build deps pre-installed, so the
# pipeline step itself does not need apt-get (and therefore does not
# need root inside the container).
#
# Build & publish (must be re-run whenever this Dockerfile changes,
# otherwise the CI lane will keep pulling the previous image):
#   docker build -f buildlib/dockers/ze-builder.Dockerfile \
#       -t ghcr.io/yuanwu2017/ucx-ze-builder:ubuntu24.04 .
#   docker push ghcr.io/yuanwu2017/ucx-ze-builder:ubuntu24.04

FROM intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -q && \
    apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        libtool \
        m4 \
        make \
        pkg-config \
        doxygen \
        ca-certificates \
        fakeroot \
        debhelper \
        dpkg-dev \
        libibverbs-dev \
        librdmacm-dev && \
    rm -rf /var/lib/apt/lists/*
