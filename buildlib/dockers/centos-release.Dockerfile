ARG CUDA_VERSION
ARG OS_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-devel-centos${OS_VERSION}

RUN yum install -y \
    autoconf \
    automake \
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
    tcl \
    tcsh \
    tk \
    wget \
    libusbx \
    fuse-libs \
    python36 \
    lsof \
    ethtool \
    environment-modules \
    valgrind-devel \
    && yum clean all

# MOFED
ARG MOFED_VERSION
ARG MOFED_OS
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
        --skip-distro-check \
        --distro ${MOFED_OS} \
    && rm -rf ${MOFED_DIR} && rm -rf *.tgz

ENV CPATH /usr/local/cuda/include:${CPATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH /usr/local/cuda/lib64:${LIBRARY_PATH}

RUN cd /usr/lib64 && \
    ln -s libudev.so.1 libudev.so && \
    ln -s libz.so.1 libz.so
