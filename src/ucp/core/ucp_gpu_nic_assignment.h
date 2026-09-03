/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_GPU_NIC_ASSIGNMENT_H_
#define UCP_GPU_NIC_ASSIGNMENT_H_

#include <ucs/datastruct/static_bitmap.h>
#include <ucs/sys/topo/base/topo_groups.h>

#include <stddef.h>
#include <stdint.h>

BEGIN_C_DECLS


#define UCP_GPU_NIC_BITMAP_INDEX_INVALID UCS_SYS_DEVICE_ID_COUNT


typedef ucs_static_bitmap_s(UCS_SYS_DEVICE_ID_COUNT)
        ucp_gpu_nic_sys_dev_bitmap_t;


typedef struct {
    ucp_gpu_nic_sys_dev_bitmap_t *nic_sys_dev_bitmaps;
    size_t                       num_bitmaps;
    uint8_t bitmap_idx_by_gpu_sys_dev[UCS_SYS_DEVICE_ID_COUNT];
} ucp_gpu_nic_assignment_t;


typedef enum {
    /**
     * Assign each run of NICs forward and then backward across the GPUs:
     * 0, 1, ..., N-1, N-1, ..., 1, 0.
     */
    UCP_GPU_NIC_ASSIGNMENT_POLICY_FLIP,

    /**
     * Assign NICs to GPUs repeatedly in ascending order:
     * 0, 1, ..., N-1, 0, 1, ...
     */
    UCP_GPU_NIC_ASSIGNMENT_POLICY_ROUND_ROBIN,

    UCP_GPU_NIC_ASSIGNMENT_POLICY_LAST
} ucp_gpu_nic_assignment_policy_t;


/**
 * Build GPU-to-NIC assignments from topology groups.
 *
 * @param [in]  groups        Topology groups to build assignments from.
 * @param [in]  policy        Assignment policy.
 * @param [out] assignment_p  Completed assignment. Updated only on success.
 *
 * @return UCS_OK on success, or an error status if assignment construction
 *         failed.
 */
ucs_status_t
ucp_gpu_nic_assignment_build(const ucs_topo_groups_t *groups,
                             ucp_gpu_nic_assignment_policy_t policy,
                             ucp_gpu_nic_assignment_t *assignment_p);


/**
 * Look up the NIC system-device bitmap assigned to a GPU system device.
 *
 * @param [in] assignment  GPU-to-NIC assignment.
 * @param [in] gpu_sys_dev GPU system device to look up.
 *
 * @return Assigned NIC system-device bitmap. The bitmap is empty if the GPU
 *         has no assigned NICs. Returns NULL if @a gpu_sys_dev is unknown or
 *         is not represented in the assignment.
 */
const ucp_gpu_nic_sys_dev_bitmap_t *
ucp_gpu_nic_assignment_lookup(const ucp_gpu_nic_assignment_t *assignment,
                              ucs_sys_device_t gpu_sys_dev);


/**
 * Test whether a network system device is present in a GPU assignment bitmap.
 *
 * @param [in] bitmap       Valid GPU-to-NIC assignment bitmap.
 * @param [in] net_sys_dev  Network system device to test.
 *
 * @return Nonzero if @a net_sys_dev is present in @a bitmap, or zero if the
 *         system device is unknown or the bit is not set.
 */
int ucp_gpu_nic_bitmap_test(const ucp_gpu_nic_sys_dev_bitmap_t *bitmap,
                            ucs_sys_device_t net_sys_dev);


/**
 * Release an assignment returned by @ref ucp_gpu_nic_assignment_build.
 *
 * @param [in] assignment  Assignment to release.
 */
void ucp_gpu_nic_assignment_release(ucp_gpu_nic_assignment_t *assignment);


END_C_DECLS

#endif
