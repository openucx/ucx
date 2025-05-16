ARG CUDA_VERSION
ARG UBUNTU_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG NV_DRIVER_VERSION
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata && \
    apt-get install -y \
        apt-file \
        automake \
        default-jdk \
        dh-make \
        fakeroot \
        g++ \
        git \
        openjdk-8-jdk \
        libcap2 \
        libnuma-dev \
        libtool \
        # Provide CUDA dependencies by libnvidia-compute*
        libnvidia-compute-${NV_DRIVER_VERSION} \
        make \
        maven \
        pkg-config \
        udev \
        wget \
        environment-modules \
    # Remove cuda-compat* from nvidia/cuda:x86_64 images, provide CUDA dependencies by libnvidia-compute* instead
    && apt-get remove -y openjdk-11-* cuda-compat* || apt-get autoremove -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# MOFED
ARG MOFED_VERSION=5.0-1.0.0.0
ARG UBUNTU_VERSION
ARG MOFED_OS=ubuntu${UBUNTU_VERSION}
ARG ARCH
ENV MOFED_DIR MLNX_OFED_LINUX-${MOFED_VERSION}-${MOFED_OS}-${ARCH}
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
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH /usr/local/cuda/lib64:${LIBRARY_PATH}
