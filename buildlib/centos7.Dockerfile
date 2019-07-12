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
    make \
    maven \
    numactl-devel \
    rdma-core-devel \
    && yum clean dbcache packages
