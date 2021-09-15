ARG CUDA_VERSION=11.2
ARG UBUNTU_VERSION=20.04
FROM gpuci/miniconda-cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata && \
    apt-get install -y \
        automake \
        dh-make \
        g++ \
        git \
        libcap2 \
        libnuma-dev \
        libtool \
        make \
        udev \
        wget \
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

# Required arguments
ARG PYTHON_VER=3.8
ARG BUILD_STACK_VER=9.3.0

# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

# Add a condarc for channels and override settings
RUN echo -e "\
auto_update_conda: False \n\
ssl_verify: False \n\
channels: \n\
  - gpuci \n\
  - nvidia \n\
  - conda-forge \n" > /opt/conda/.condarc \
      && cat /opt/conda/.condarc

# Update and add pkgs for gpuci builds
RUN apt-get update -y --fix-missing \
    && apt-get -qq install apt-utils -y --no-install-recommends \
    && apt-get install -y \
      chrpath \
      file \
      screen \
      tzdata \
      vim \
      zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get -y upgrade \
    && rm -rf /var/lib/apt/lists/*

# Add core tools to base env
RUN conda install -y gpuci-tools \
    || conda install -y gpuci-tools

# Create `ucx-py` conda env and make it default
RUN gpuci_conda_retry create --no-default-packages --override-channels -n ucx-py \
      -c nvidia \
      -c conda-forge \
      -c gpuci \
      cudatoolkit=${CUDA_VERSION} \
      git \
      gpuci-tools \
      libgcc-ng=${BUILD_STACK_VER} \
      libstdcxx-ng=${BUILD_STACK_VER} \
      python=${PYTHON_VER} \
      'python_abi=*=*cp*' \
      "setuptools<50" \
    && sed -i 's/conda activate base/conda activate ucx-py/g' ~/.bashrc

RUN source activate ucx-py \
  && env \
  && conda info \
  && conda config --show-sources \
  && conda list --show-channel-urls

RUN chmod -R ugo+w /opt/conda \
  && conda clean -tipy \
  && chmod -R ugo+w /opt/conda
