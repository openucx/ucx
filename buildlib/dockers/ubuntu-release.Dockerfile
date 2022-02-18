ARG CUDA_VERSION=10.1
ARG UBUNTU_VERSION=16.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata && \
    apt-get install -y \
        automake \
        default-jdk \
        dh-make \
        g++ \
        git \
        openjdk-8-jdk \
        libcap2 \
        libnuma-dev \
        libtool \
        make \
        maven \
        udev \
        wget \
        environment-modules \
    && apt-get remove -y openjdk-11-* || apt-get autoremove -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# MOFED
ARG MOFED_VERSION=5.0-1.0.0.0
ARG UBUNTU_VERSION
ARG MOFED_OS=ubuntu${UBUNTU_VERSION}
ENV MOFED_DIR MLNX_OFED_LINUX-${MOFED_VERSION}-${MOFED_OS}-x86_64
ENV MOFED_SITE_PLACE MLNX_OFED-${MOFED_VERSION}
ENV MOFED_IMAGE ${MOFED_DIR}.tgz
RUN wget --no-verbose http://content.mellanox.com/ofed/${MOFED_SITE_PLACE}/${MOFED_IMAGE} && \
    tar -xzf ${MOFED_IMAGE}
RUN ${MOFED_DIR}/mlnxofedinstall --all -q \
        --user-space-only \
        --without-fw-update \
        --skip-distro-check \
        --without-ucx \
        --without-hcoll \
        --without-openmpi \
        --without-sharp && \
    rm -rf ${MOFED_DIR} && rm -rf *.tgz

ENV CPATH /usr/local/cuda/include:${CPATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/compat:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/compat:${LIBRARY_PATH}
ENV PATH /usr/local/cuda/compat:${PATH}

RUN ml_stub=$(find /usr -name libnvidia-ml.so) && ln -s $ml_stub /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
