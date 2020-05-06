# docker build -t ucfconsort.azurecr.io/ucx/fedora:1 -f buildlib/fedora.Dockerfile buildlib/
FROM fedora:30

RUN dnf install -y \
    autoconf \
    automake \
    clang \
    cppcheck \
    csclng \
    cscppc \
    csmock-common \
    doxygen \
    file \
    gcc-c++ \
    git \
    glibc-devel \
    java-latest-openjdk-devel \
    libtool \
    make \
    maven \
    numactl-devel \
    rdma-core-devel \
    rpm-build \
    && (dnf remove -y java-1.8.0-openjdk-devel || true) \
    && dnf clean dbcache packages
