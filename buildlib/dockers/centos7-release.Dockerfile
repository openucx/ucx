ARG CUDA_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-devel-centos7

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
    tcl \
    tcsh \
    tk \
    wget \
    libusbx \
    fuse-libs \
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
    && rm -rf ${MOFED_DIR} && rm -rf *.tgz

ENV CPATH /usr/local/cuda/include:${CPATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/compat:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/compat:${LIBRARY_PATH}
ENV PATH /usr/local/cuda/compat:${PATH}
