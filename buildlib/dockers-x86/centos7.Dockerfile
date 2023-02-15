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
    zlib-devel \
    libudev-devel \
    valgrind-devel \
    environment-modules \
    make \
    maven \
    numactl-devel \
    rdma-core-devel \
    rpm-build \
    tcl \
    tcsh \
    tk \
    wget \
    libusbx \
    fuse-libs \
    lsof \
    ethtool \
    glibc-static \
    && yum clean all

ARG MOFED_OS=rhel7.6
ARG MOFED_VERSION=5.0-1.0.0.0
ENV MOFED_DIR MLNX_OFED_LINUX-${MOFED_VERSION}-${MOFED_OS}-x86_64
ENV MOFED_SITE_PLACE MLNX_OFED-${MOFED_VERSION}
ENV MOFED_IMAGE ${MOFED_DIR}.tgz

RUN wget --no-verbose http://content.mellanox.com/ofed/${MOFED_SITE_PLACE}/${MOFED_IMAGE} && \
    tar -xzf ${MOFED_IMAGE} && \
    ${MOFED_DIR}/mlnxofedinstall --all -q \
        --user-space-only \
        --without-fw-update \
        --skip-distro-check \
        --without-ucx \
        --without-hcoll \
        --without-openmpi \
        --without-sharp \
        --distro ${MOFED_OS} \
    && rm -rf ${MOFED_DIR} && rm -rf *.tgz
