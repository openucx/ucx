#!/bin/sh -eE
#
# Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

#
# This script detects potentially stuck Queue Pairs which have ci != pi.
#

VFS_UCX_PATH="/tmp/ucx"
HW_CI_FILE="hw_ci"
PREV_SW_PI_FILE="prev_sw_pi"
QP_NUM_FILE="qp_num"

# Interval between traversing QPs (in seconds)
QP_CHECK_INTERVAL=10
# "yes" - print all QPs, "no" - print only stuck QPs
PRINT_ALL_QPS=0

# Show script usage help message
usage()
{
    echo " Usage:"
    echo "  "$0" [ options ]"
    echo " Options:"
    echo "  -a           Print all QPs"
    echo "  -p <path>    Path to UCX VFS mount point (default: ${VFS_UCX_PATH})"
    echo "  -i <seconds> Interval to check QP state"
    exit 1
}

while getopts ":ap:i:" o; do
    case "${o}" in
    a)
        PRINT_ALL_QPS=1
        ;;
    i)
        QP_CHECK_INTERVAL=${OPTARG}
        ;;
    p)
        VFS_UCX_PATH=${OPTARG}
        ;;
    *)
        usage
        ;;
    esac
done
shift $((OPTIND-1))

declare -A qp_nums
declare -A initial_hw_cis

traverse() {
    DC_TXWQ_GLOB_PATH="${VFS_UCX_PATH}/*/uct/worker/*/iface/*/dci_pool/*/*"
    RC_EP_GLOB_PATH="${VFS_UCX_PATH}/*/uct/worker/*/iface/*/ep/*"
    for file in ${DC_TXWQ_GLOB_PATH}/${QP_NUM_FILE} ${RC_EP_GLOB_PATH}/${QP_NUM_FILE}
    do
        filename=$(basename ${file})
        dir=$(dirname ${file})
        if [ -f ${dir}/${HW_CI_FILE} ] && \
            [ -f ${dir}/${PREV_SW_PI_FILE} ] ; then
            qp_num=$(<${file})

            if [ ! ${qp_nums[${qp_num}]} ] ; then
                qp_nums[${qp_num}]=${dir}
                initial_hw_cis[${qp_num}]=$(<${dir}/${HW_CI_FILE})
            fi
        fi
    done
}

print_qp_num_info() {
    for qp_num in "${!qp_nums[@]}"
    do
        dir=${qp_nums[${qp_num}]}
        if [ ! -d ${dir} ] ; then
            continue
        fi

        hw_ci=$(<${dir}/${HW_CI_FILE})
        prev_sw_pi=$(<${dir}/${PREV_SW_PI_FILE})
        initial_hw_ci=${initial_hw_cis[${qp_num}]}

        # QP is considered as stuck if (hw_ci != sw_pi) AND hw_ci hasn't been changed
        if [ ${hw_ci} -eq ${prev_sw_pi} ] || [ ${initial_hw_ci} -ne ${hw_ci} ] ; then
            result_str="ok"
            if [ ${PRINT_ALL_QPS} -eq 0 ] ; then
                continue
            fi
        else
            result_str="stuck (path=$dir)"
        fi

        echo "qp=0x${qp_num}: pi=${prev_sw_pi} ci=${hw_ci} initial_ci=${initial_hw_ci} - ${result_str}"
    done
}

traverse
sleep ${QP_CHECK_INTERVAL}
traverse

print_qp_num_info
