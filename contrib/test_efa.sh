#!/usr/bin/bash -eEx
#
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

source $(dirname $0)/../buildlib/az-helpers.sh
source $(dirname $0)/../buildlib/tools/common.sh

WORKSPACE=${WORKSPACE:=$PWD}
ucx_inst=${WORKSPACE}/install

if [ ! -f ./autogen.sh ]
then
    echo "error: run from UCX root"
    exit 1
fi

rdma_core_version=55
rdma_core_tarball=v$rdma_core_version.0.tar.gz
rdma_core=../rdma-core-$rdma_core_version.0
rdma_core_tarball_sha=a02d128974055ffa92577e4d3889213ac180a79f05b077aeb884bafb6b46e957

build_rdma_core_efa() {
    # Fetch rdma-core
    if [ ! -d $rdma_core ]
    then
        wget -O ../$rdma_core_tarball \
            https://github.com/linux-rdma/rdma-core/archive/refs/tags/$rdma_core_tarball
        echo "$rdma_core_tarball_sha ../$rdma_core_tarball" | sha256sum --check
        tar zxvf ../$rdma_core_tarball -C ../
    fi

    # Build rdma-core
    cd $rdma_core
    ./build.sh
    ls -alrt ./build/include/infiniband/
    cd -

    # Build IBMOCK
    cd contrib/ibmock
    make INCLUDES=-I"$(pwd)/../../$rdma_core/build/include"
    cd -
}

build_ucx_efa() {
    # Build UCX with EFA
    [ -f Makefile ] && make distclean
    ./autogen.sh
    cd contrib
    build devel --enable-gtest \
        --with-verbs="$(pwd)/../$rdma_core/build" --with-efa --without-mlx5 \
        --without-knem --without-gdrcopy --without-cuda
    cd -
}

build_rdma_core_rpm() {
    sudo yum -y install ninja-build pandoc perl-generators python3-Cython python3-docutils
    cd $rdma_core
    rm -rf SOURCES tmp || :
    mkdir SOURCES tmp
    tar --wildcards -xzf ../$rdma_core_tarball $(basename $rdma_core)/redhat/rdma-core.spec --strip-components=2
    RPM_SRC=$((rpmspec -P *.spec || grep ^Source: *.spec) | awk '/^Source:/{split($0,a,"[ \t]+");print(a[2])}')
    (cd SOURCES && ln -sf ../../$rdma_core_tarball "$RPM_SRC")
    rpmbuild --define '_tmppath '$(pwd)'/tmp' --define '_topdir '$(pwd) -bb *.spec
    cd -
}

install_rdma_core_rpm() {
    local pkgs="libibumad infiniband-diags libibverbs librdmacm rdma-core rdma-core-devel"
    for pkg in $pkgs; do
        sudo yum -y remove $pkg || :
    done

    for pkg in $pkgs; do
        sudo rpm -i $rdma_core/RPMS/x86_64/$pkg-$rdma_core_version*.rpm
    done
}

install_rdma_core() {
    cd $rdma_core/build
    sudo /bin/cp -r -f lib/lib*efa*.so* /usr/lib64/
    sudo /bin/cp -r -f include/infiniband/*efa*h /usr/include/infiniband/
    cd -
}

run_gtests() {
    if [ ! -d /dev/infiniband ]
    then
        mkdir /dev/infiniband
        mknod /dev/infiniband/uverbs0 c 1 3
        mknod /dev/infiniband/uverbs1 c 1 3
        chmod ugo+rw /dev/infiniband/uver*
    fi

    # Setup ibmock library
    export LD_LIBRARY_PATH=$(pwd)/contrib/ibmock/build:$(pwd)/$rdma_core/build/lib:$LD_LIBRARY_PATH
    export CLOUD_TYPE=aws

    # Check basic ibmock functionality, error if mocked interface is not found
    UCX_LOG_LEVEL=trace \
    UCX_PROTO_INFO=y UCX_TLS=ud UCX_NET_DEVICES=rdmap0:1 \
        ./install/bin/ucx_perftest -l -t tag_bw
    ./install/bin/ucx_info -d

    # Try the faster approach before valgrind
    make -C contrib/test/gtest test GTEST_FILTER=*ud*:*test_srd*
    make -C contrib/test/gtest test_valgrind GTEST_FILTER=*ud*:*test_srd*:-*test_uct_perf.envelope*
}

test_ucx_rpm() {
    cd contrib
    ./buildrpm.sh -s -t -b
    rpm -qplvi rpm-dist/*/ucx-ib-efa-[1-9]*rpm | grep libuct_ib_efa.so
    rpm -qplvi rpm-dist/*/ucx-ib-efa-[1-9]*rpm
    cd -
}

case "${1-:}" in
    install_rdma_core_efa)
        install_rdma_core
        ;;
    test_rpm_efa)
        build_ucx_efa
        test_ucx_rpm
        ;;
    build_efa)
        build_rdma_core_efa
        build_ucx_efa
        ;;
    gtest_efa)
        run_gtests
        ;;
    *)
        echo "error: invalid argument"
        exit 1
esac
