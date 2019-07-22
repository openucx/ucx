# docker build -t ucfconsort.azurecr.io/ucx/fedora:1 -f buildlib/fedora.Dockerfile buildlib/
FROM fedora:30

RUN dnf install -y \
    autoconf \
    automake \
    clang \
    cppcheck \
    cscppc \
    csmock-common \
    doxygen \
    file \
    gcc-c++ \
    git \
    glibc-devel \
    libtool \
    make \
    maven \
    numactl-devel \
    rdma-core-devel \
    rpm-build \
    && dnf clean dbcache packages
