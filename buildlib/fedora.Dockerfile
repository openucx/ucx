# docker build -t ucfconsort.azurecr.io/ucx/fedora:5 -f buildlib/fedora.Dockerfile buildlib/
FROM fedora:33

RUN dnf install -y \
    autoconf \
    automake \
    cmake \
    cppcheck \
    csclng \
    cscppc \
    csmock-common \
    doxygen \
    file \
    gcc-c++ \
    git \
    git-clang-format \
    glibc-devel \
    java-1.8.0-openjdk-devel \
    libtool \
    make \
    maven \
    numactl-devel \
    python \
    rdma-core-devel \
    rpm-build \
    && dnf clean dbcache packages
RUN export BUILD_ROOT=/tmp/llvm-project && \
    git clone https://github.com/openucx/llvm-project.git --depth=1 -b ucx-clang-format --single-branch ${BUILD_ROOT} && \
    mkdir -p ${BUILD_ROOT}/build && cd ${BUILD_ROOT}/build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=clang -G "Unix Makefiles" \
    ../llvm && \
    make -j$(nproc) && make install && rm -rf ${BUILD_ROOT}
