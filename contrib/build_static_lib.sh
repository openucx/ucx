#!/bin/bash -eE
#
# Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

# This scripts creates a single static UCX library, which also includes
# ibverbs and ibcm libs.

usage() {
    echo "Usage: $0 [option...]"
    echo "Builds a single static UCX library with ibverbs and ibcm included"
    echo
    echo "   -u, --ucx <PATH>          UCX installation prefix"
    echo "   -i, --ib  <PATH>          infiniband verbs installation prefix (/usr)"
    echo
    exit 1
}

while [ "$1" != "" ]; do
key="$1"
case $key in
    -u|--ucx)
    [ -z $2 ] && usage
    UCX_PREFIX="$2"
    ;;
    -i|--ib)
    [ -z $2 ] && usage
    IB_PREFIX="$2"
    ;;
    *)
    usage
    ;;
esac
shift 2
done

# UCX prefix is required, IB prefix is /usr by default
[ -z $UCX_PREFIX ] && usage
IB_PREFIX="${IB_PREFIX:-/usr}"

[ -d "${IB_PREFIX}/lib64" ] && ib_lib_suffix="64"

UCX_LIB_PATH="${UCX_PREFIX}/lib"
IB_LIB_PATH="${IB_PREFIX}/lib${ib_lib_suffix}"

check_lib_dirs() {
    if [ ! -d "$1" ]; then
        echo "$1 does not exist"
        exit 1
    fi
}

check_libs() {
    lib_path=$1 && shift
    for lib in "$@"; do
        lib_file="${lib_path}/lib${lib}.a"
        if [ ! -f $lib_file ]; then
            echo "$lib_file is missing"
            exit 1
        fi
    done

}

# Check that all needed libs are present
check_lib_dirs $UCX_PREFIX
check_lib_dirs $IB_PREFIX
check_libs $UCX_LIB_PATH ucp uct ucs ucm
check_libs $IB_LIB_PATH ibverbs ibcm

# Check that UCX_LIB_PATH is writable
if [ ! -w ${UCX_LIB_PATH} ]; then
    echo "Permissions denied, can't write to ${UCX_LIB_PATH}"
    exit 1
fi

ar -M <<EOM
CREATE ${UCX_LIB_PATH}/libucx.a
ADDLIB ${UCX_LIB_PATH}/libucp.a
ADDLIB ${UCX_LIB_PATH}/libuct.a
ADDLIB ${IB_LIB_PATH}/libibverbs.a
ADDLIB ${IB_LIB_PATH}/libibcm.a
ADDLIB ${UCX_LIB_PATH}/libucs.a
ADDLIB ${UCX_LIB_PATH}/libucm.a
SAVE
END
EOM

