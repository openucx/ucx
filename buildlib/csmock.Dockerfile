# docker build -t ucfconsort.azurecr.io/ucx/csmock:latest -f buildlib/csmock.Dockerfile buildlib/
FROM fedora:30

RUN dnf install -y \
    autoconf \
    automake \
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
    csmock \
    && dnf clean dbcache packages

RUN useradd -m -u 1001 vsts_azpcontainer &&\
    usermod -aG mock vsts_azpcontainer
