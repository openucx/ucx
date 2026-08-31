/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ucp_gpu_nic_assignment.h"

#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>

#include <string.h>


typedef struct {
    size_t gpu_idx;
    int    direction; /* 1 for forward, -1 for backward */
} ucp_gpu_nic_policy_state_t;


static void ucp_gpu_nic_policy_advance(ucp_gpu_nic_policy_t policy,
                                       size_t num_gpus,
                                       ucp_gpu_nic_policy_state_t *state)
{
    if (num_gpus == 1) {
        return;
    }

    if (policy == UCP_GPU_NIC_POLICY_FLIP) {
        if (state->direction > 0) {
            if (state->gpu_idx == (num_gpus - 1)) {
                /* change direction */
                state->direction = -1;
            } else {
                /* keep going forward */
                ++state->gpu_idx;
            }
        } else {
            if (state->gpu_idx == 0) {
                /* change direction */
                state->direction = 1;
            } else {
                /* keep going backward */
                --state->gpu_idx;
            }
        }
    } else if (policy == UCP_GPU_NIC_POLICY_ALT) {
        state->gpu_idx = (state->gpu_idx + 1) % num_gpus;
    }
}

static void ucp_gpu_nic_assignment_build_sys_dev_bitmap(
        const ucs_topo_group_t *group, size_t target_gpu_idx,
        ucp_gpu_nic_policy_t policy,
        ucp_gpu_nic_sys_dev_bitmap_t *nic_sys_dev_bitmap)
{
    ucp_gpu_nic_policy_state_t state = {
        .gpu_idx   = 0,
        .direction = 1
    };
    const ucs_topo_nic_t *nic;
    const ucs_sys_device_t *port;
    size_t num_gpus;

    num_gpus = ucs_array_length(&group->gpus);
    ucs_assert(num_gpus > 0);
    UCS_STATIC_BITMAP_RESET_ALL(nic_sys_dev_bitmap);

    ucs_array_for_each(nic, &group->nics) {
        ucs_assert((nic->num_ports > 0) &&
                   (nic->num_ports <= UCS_TOPO_MAX_PORTS_PER_NIC));

        if (state.gpu_idx == target_gpu_idx) {
            ucs_carray_for_each(port, nic->ports, nic->num_ports) {
                ucs_assert(*port != UCS_SYS_DEVICE_ID_UNKNOWN);
                UCS_STATIC_BITMAP_SET(nic_sys_dev_bitmap, *port);
            }
        }

        ucp_gpu_nic_policy_advance(policy, num_gpus, &state);
    }
}

static void ucp_gpu_nic_assignment_init(ucp_gpu_nic_assignment_t *assignment)
{
    ucs_array_init_dynamic(&assignment->nic_sys_dev_bitmaps);
    memset(assignment->bitmap_idx_by_gpu_sys_dev,
           UCP_GPU_NIC_BITMAP_INDEX_INVALID,
           sizeof(assignment->bitmap_idx_by_gpu_sys_dev));
}

static ucs_status_t ucp_gpu_nic_assignment_add_gpu(
        ucp_gpu_nic_assignment_t *assignment, const ucs_topo_gpu_t *gpu,
        const ucp_gpu_nic_sys_dev_bitmap_t *nic_sys_dev_bitmap)
{
    ucp_gpu_nic_sys_dev_bitmap_t *nic_sys_dev_bitmap_p;
    const ucs_sys_device_t *gpu_sys_dev;
    uint8_t nic_sys_dev_bitmap_idx;

    if (ucs_array_length(&assignment->nic_sys_dev_bitmaps) >=
        UCP_GPU_NIC_BITMAP_INDEX_INVALID) {
        ucs_error("gpu-nic sys-dev bitmap count exceeds limit %u",
                  (unsigned)UCP_GPU_NIC_BITMAP_INDEX_INVALID);
        return UCS_ERR_EXCEEDS_LIMIT;
    }

    nic_sys_dev_bitmap_idx = (uint8_t)ucs_array_length(
            &assignment->nic_sys_dev_bitmaps);
    nic_sys_dev_bitmap_p   = ucs_array_append(
            &assignment->nic_sys_dev_bitmaps,
            ucs_error("failed to allocate gpu-nic sys-dev bitmap");
            return UCS_ERR_NO_MEMORY);
    *nic_sys_dev_bitmap_p  = *nic_sys_dev_bitmap;

    ucs_carray_for_each(gpu_sys_dev, gpu->devices, gpu->num_devices) {
        ucs_assert(*gpu_sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN);
        ucs_assert(assignment->bitmap_idx_by_gpu_sys_dev[*gpu_sys_dev] ==
                   UCP_GPU_NIC_BITMAP_INDEX_INVALID);

        assignment->bitmap_idx_by_gpu_sys_dev[*gpu_sys_dev] =
                nic_sys_dev_bitmap_idx;
    }

    return UCS_OK;
}

ucs_status_t
ucp_gpu_nic_assignment_build(const ucs_topo_groups_t *groups,
                             ucp_gpu_nic_policy_t policy,
                             ucp_gpu_nic_assignment_t *assignment_p)
{
    ucp_gpu_nic_sys_dev_bitmap_t nic_sys_dev_bitmap =
            UCS_STATIC_BITMAP_ZERO_INITIALIZER;
    ucp_gpu_nic_assignment_t assignment;
    const ucs_topo_group_t *group;
    const ucs_topo_gpu_t *gpu;
    ucs_status_t status;
    size_t gpu_idx;

    ucs_assert(groups != NULL);
    ucs_assertv((policy >= 0) && (policy < UCP_GPU_NIC_POLICY_LAST),
                "invalid gpu-nic assignment policy %d", policy);

    ucp_gpu_nic_assignment_init(&assignment);

    ucs_array_for_each(group, &groups->groups) {
        if (ucs_array_is_empty(&group->nics) ||
            ucs_array_is_empty(&group->gpus)) {
            ucs_debug("group %p has %zu gpus and %zu nics, skipping", group,
                      ucs_array_length(&group->gpus),
                      ucs_array_length(&group->nics));
            continue;
        }

        for (gpu_idx = 0; gpu_idx < ucs_array_length(&group->gpus); ++gpu_idx) {
            gpu = &ucs_array_elem(&group->gpus, gpu_idx);
            ucs_assert((gpu->num_devices > 0) &&
                       (gpu->num_devices <= UCS_TOPO_MAX_DEVICES_PER_GPU));

            ucp_gpu_nic_assignment_build_sys_dev_bitmap(group, gpu_idx, policy,
                                                        &nic_sys_dev_bitmap);
            if (UCS_STATIC_BITMAP_IS_ZERO(nic_sys_dev_bitmap)) {
                continue;
            }

            status = ucp_gpu_nic_assignment_add_gpu(&assignment, gpu,
                                                    &nic_sys_dev_bitmap);
            if (status != UCS_OK) {
                goto err_cleanup_sys_dev_bitmaps;
            }
        }
    }

    ucs_assert(assignment_p != NULL);
    *assignment_p = assignment;

    return UCS_OK;

err_cleanup_sys_dev_bitmaps:
    ucs_array_cleanup_dynamic(&assignment.nic_sys_dev_bitmaps);
    return status;
}

const ucp_gpu_nic_sys_dev_bitmap_t *
ucp_gpu_nic_assignment_lookup(const ucp_gpu_nic_assignment_t *assignment,
                              ucs_sys_device_t gpu_sys_dev)
{
    uint8_t nic_sys_dev_bitmap_idx;

    ucs_assert(assignment != NULL);

    if (gpu_sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
        return NULL;
    }

    nic_sys_dev_bitmap_idx = assignment->bitmap_idx_by_gpu_sys_dev[gpu_sys_dev];
    if (nic_sys_dev_bitmap_idx == UCP_GPU_NIC_BITMAP_INDEX_INVALID) {
        return NULL;
    }

    ucs_assert(nic_sys_dev_bitmap_idx <
               ucs_array_length(&assignment->nic_sys_dev_bitmaps));
    return &ucs_array_elem(&assignment->nic_sys_dev_bitmaps,
                           nic_sys_dev_bitmap_idx);
}

int ucp_gpu_nic_bitmap_test(const ucp_gpu_nic_sys_dev_bitmap_t *bitmap,
                            ucs_sys_device_t net_sys_dev)
{
    ucs_assert(bitmap != NULL);

    if (net_sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
        return 0;
    }

    return UCS_STATIC_BITMAP_GET(*bitmap, net_sys_dev);
}

void ucp_gpu_nic_assignment_release(ucp_gpu_nic_assignment_t *assignment)
{
    ucs_array_cleanup_dynamic(&assignment->nic_sys_dev_bitmaps);
}
