# docker build -t ucfconsort.azurecr.io/ucx/centos7:1 -f buildlib/centos7.Dockerfile buildlib/
FROM centos:7

RUN yum install -y \
    autoconf \
    automake \
    doxygen \
    file \
    gcc-c++ \
    git \
    glibc-devel \
    libtool \
    librdmacm \
    make \
    maven \
    numactl-devel \
    rdma-core-devel \
    rpm-build \
    && yum clean dbcache packages
