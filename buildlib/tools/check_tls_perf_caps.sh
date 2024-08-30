#!/bin/bash -eE
#
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See file LICENSE for terms.
#

realdir="$(realpath "$(dirname "$0")")"
source "${realdir}"/../az-helpers.sh

# check that correct UCX versions are used
LD_LIBRARY_PATH=${UCX_LEGACY_PATH}/lib "${UCX_LEGACY_PATH}"/bin/ucx_info -v
LD_LIBRARY_PATH=${UCX_PR_PATH}/lib "${UCX_PR_PATH}"/bin/ucx_info -v

LD_LIBRARY_PATH=${UCX_LEGACY_PATH}/lib old_out=$("${UCX_LEGACY_PATH}"/bin/ucx_info -d)
LD_LIBRARY_PATH=${UCX_PR_PATH}/lib new_out=$("${UCX_PR_PATH}"/bin/ucx_info -d)

export UCX_TCP_BRIDGE_ENABLE=y #bridge devices are not shown by default since v1.16

res=true
for tl_name in $(echo "${old_out}" | grep Transport | awk '{print $3}')
do
    old_tl_caps=$(echo "$old_out" | grep -A 8 "Transport: $tl_name")
    new_tl_caps=$(echo "$new_out" | grep -A 8 "Transport: $tl_name")
    for device in $(echo "${old_tl_caps}" | grep Device | awk '{print $3}')
    do
      old_caps=$(echo "$old_tl_caps" | grep -A 7 "Device: $device")
      new_caps=$(echo "$new_tl_caps" | grep -A 7 "Device: $device")
      for cap in bandwidth latency overhead
      do
        old_cap=$(echo "$old_caps" | grep $cap | sed -e "s/^[^:]*:[ \t]*//")
        new_cap=$(echo "$new_caps" | grep $cap | sed -e "s/^[^:]*:[ \t]*//")
        if [ "$old_cap" != "$new_cap" ]
        then
          azure_log_error "Fail: (${device}/${tl_name}/${cap}) ${old_cap} != ${new_cap}"
          res=false
        else
          echo "${device}/${tl_name}/${cap}: ${old_cap}"
        fi
       done
    done
done

$res
