#!/bin/bash -eE
#
# Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

# This scripts creates a single static UCX library, which also includes
# ibverbs and ibcm libs (if present).

usage() {
    echo "Usage: $0 [option...]"
    echo "Builds a single static UCX library with ibverbs and ibcm included (if needed)"
    echo
    echo "   -u, --ucx <PATH>          UCX installation prefix"
    echo "   -i, --ib  <PATH>          infiniband verbs installation prefix (/usr)"
    echo "   -o, --out <PATH>          Path where to store built library ({UCX_PATH}/lib)"
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
    -o|--out)
    [ -z $2 ] && usage
    OUT_DIR="$2"
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
OUT_DIR="${OUT_DIR:-${UCX_LIB_PATH}}"

check_dir() {
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

# Check that libcm is needed for UCT

if grep -q libcm $UCX_LIB_PATH/libuct.la; then
    ib_cm_lib="ibcm"
    ib_cm_lib_str="ADDLIB ${IB_LIB_PATH}/libibcm.a"
fi

if grep -q libverbs $UCX_LIB_PATH/libuct.la; then
    ib_verbs_lib="ibverbs"
    ib_verbs_lib_str="ADDLIB ${IB_LIB_PATH}/libibverbs.a"
fi

# Check that all needed libs are present
check_dir $UCX_PREFIX
check_dir $IB_PREFIX
check_dir $OUT_DIR
check_libs $UCX_LIB_PATH ucp uct ucs ucm
check_libs $IB_LIB_PATH $ib_verbs_lib $ib_cm_lib

# Check that UCX_LIB_PATH is writable
if [ ! -w ${OUT_DIR} ]; then
    echo "Permissions denied, can't write to ${OUT_DIR}"
    exit 1
fi

ar -M <<EOM
CREATE ${OUT_DIR}/libucx.a
ADDLIB ${UCX_LIB_PATH}/libucp.a
ADDLIB ${UCX_LIB_PATH}/libuct.a
${ib_verbs_lib_str}
${ib_cm_lib_str}
ADDLIB ${UCX_LIB_PATH}/libucs.a
ADDLIB ${UCX_LIB_PATH}/libucm.a
SAVE
END
EOM

