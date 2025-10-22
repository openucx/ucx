ARG CUDA_VERSION
ARG OS_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-devel-rockylinux${OS_VERSION}

ARG MOFED_VERSION
ARG MOFED_OS
ARG ARCH

RUN yum install -y \
    autoconf \
    automake \
    environment-modules \
    ethtool \
    file \
    fuse-libs \
    gcc-c++ \
    git \
    glibc-devel \
    libtool \
    libusbx \
    lsof \
    make \
    maven \
    numactl-devel \
    pciutils \
    perl \
    pinentry \
    python3 \
    rdma-core-devel \
    rpm-build \
    tcl \
    tcsh \
    tk \
    valgrind-devel \
    wget \
    && yum clean all \
    && rm -rf /var/cache/yum

ENV MOFED_DIR=MLNX_OFED_LINUX-${MOFED_VERSION}-${MOFED_OS}-${ARCH} \
    MOFED_SITE_PLACE=MLNX_OFED-${MOFED_VERSION} \
    CPATH=/usr/local/cuda/include:${CPATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH}

RUN wget --no-verbose http://content.mellanox.com/ofed/${MOFED_SITE_PLACE}/${MOFED_DIR}.tgz \
    && tar -xzf ${MOFED_DIR}.tgz \
    && ${MOFED_DIR}/mlnxofedinstall --basic -q \
        --user-space-only \
        --without-fw-update \
        --skip-distro-check \
        --without-ucx \
        --without-hcoll \
        --without-openmpi \
        --without-sharp \
        --distro ${MOFED_OS} \
    # MOFED sets memlock unlimited (required for RDMA runtime), but this breaks su in
    # unprivileged containers. Safe to remove for CI build containers.
    && sed -i '/memlock/d' /etc/security/limits.conf \
    && rm -rf ${MOFED_DIR} *.tgz \
    && cd /usr/lib64 \
    && ln -s libudev.so.1 libudev.so \
    && ln -s libz.so.1 libz.so
