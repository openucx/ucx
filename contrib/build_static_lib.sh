#!/bin/bash
#
# Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

# This scripts creates a single static UCX library, which also includes
# ibverbs and ibcm libs.

UCX="../install"
IB="/usr"

usage() {
    echo "Usage: $0 [option...]"
    echo
    echo "   -u, --ucx <PATH>          use UCX libs from the given path"
    echo "   -i, --ib  <PATH>          use IB verbs libs from the given path"
    echo
    exit 1
}

while [ "$1" != "" ]; do
key="$1"
case $key in
    -u|--ucx)
    [ -z $2 ] && usage
    UCX="$2"
    ;;
    -i|--ib)
    [ -z $2 ] && usage
    IB="$2"
    ;;
    *)
    usage
    ;;
esac
shift 2
done

check_lib_dirs() {
    if [ ! -d "$1" ] || ! [ -d "${1}/lib" -o -d "${1}/lib64" ]; then
        echo "$1 does not exist or no libs found"
        exit 1
    fi
}

check_lib_dirs $UCX
check_lib_dirs $IB

[ -d "${IB}/lib64" ] && ib_lib_suffix="64"

UCX_LIB="${UCX}/lib"
IB_LIB="${IB}/lib${ib_lib_suffix}"

ar -M <<EOM
CREATE ${UCX_LIB}/libucx.a
ADDLIB ${UCX_LIB}/libucp.a
ADDLIB ${UCX_LIB}/libuct.a
ADDLIB ${IB_LIB}/libibverbs.a
ADDLIB ${IB_LIB}/libibcm.a
ADDLIB ${UCX_LIB}/libucs.a
ADDLIB ${UCX_LIB}/libucm.a
SAVE
END
EOM

