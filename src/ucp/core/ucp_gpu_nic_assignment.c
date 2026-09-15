/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ucp_gpu_nic_assignment.h"

#include <ucs/datastruct/string_buffer.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>

#include <string.h>


static size_t
ucp_gpu_nic_assignment_get_gpu_idx(ucp_gpu_nic_assignment_policy_t policy,
                                   size_t num_gpus, size_t nic_idx)
{
    size_t gpu_idx, leg;

    ucs_assert(num_gpus > 0);

    gpu_idx = nic_idx % num_gpus;
    if (policy == UCP_GPU_NIC_ASSIGNMENT_POLICY_ROUND_ROBIN) {
        return gpu_idx;
    }

    ucs_assertv(policy == UCP_GPU_NIC_ASSIGNMENT_POLICY_FLIP,
                "invalid policy: %d", (int)policy);
    leg = nic_idx / num_gpus;
    if ((leg % 2) != 0) {
        gpu_idx = num_gpus - 1 - gpu_idx;
    }

    return gpu_idx;
}

static void
ucp_gpu_nic_bitmap_add_nic(ucp_gpu_nic_sys_dev_bitmap_t *nic_sys_dev_bitmap,
                           const ucs_topo_group_element_t *nic)
{
    const ucs_sys_device_t *nic_sys_dev;

    ucs_assertv((nic->num_sys_devs > 0) && (nic->num_sys_devs <=
                                            UCS_TOPO_MAX_SYS_DEVS_PER_ELEMENT),
                "invalid num_sys_devs: %zu", nic->num_sys_devs);

    ucs_carray_for_each(nic_sys_dev, nic->sys_devs, nic->num_sys_devs) {
        ucs_assert(*nic_sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN);
        UCS_STATIC_BITMAP_SET(nic_sys_dev_bitmap, *nic_sys_dev);
    }
}

static ucp_gpu_nic_sys_dev_bitmap_t *
ucp_gpu_nic_assignment_lookup_mut(const ucp_gpu_nic_assignment_t *assignment,
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

    ucs_assertv(nic_sys_dev_bitmap_idx < assignment->num_bitmaps,
                "invalid nic_sys_dev_bitmap_idx %u (num_bitmaps: %zu) assigned "
                "to gpu_sys_dev %u",
                nic_sys_dev_bitmap_idx, assignment->num_bitmaps, gpu_sys_dev);

    return &assignment->nic_sys_dev_bitmaps[nic_sys_dev_bitmap_idx];
}

const ucp_gpu_nic_sys_dev_bitmap_t *
ucp_gpu_nic_assignment_lookup(const ucp_gpu_nic_assignment_t *assignment,
                              ucs_sys_device_t gpu_sys_dev)
{
    return ucp_gpu_nic_assignment_lookup_mut(assignment, gpu_sys_dev);
}

/* Log the assignment for a group.
 * For example:
 *  `group #0 (2 gpus, 4 nics) assignment: [ 0, 1, 1, 0 ]`
 *  `group #1 (2 gpus, 4 nics) assignment: [ 0/1, X, X, 0/1 ]`
 */
static void
ucp_gpu_nic_assignment_log_group(const ucp_gpu_nic_assignment_t *assignment,
                                 const ucs_topo_group_t *group,
                                 size_t group_idx)
{
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    size_t num_gpus          = ucs_array_length(&group->gpus);
    size_t num_nics          = ucs_array_length(&group->nics);
    const ucp_gpu_nic_sys_dev_bitmap_t *nic_sys_dev_bitmap;
    const ucs_topo_group_element_t *gpu;
    const ucs_topo_group_element_t *nic;
    size_t nic_idx;
    size_t gpu_idx;
    int has_owner;

    if (!ucs_log_is_enabled(UCS_LOG_LEVEL_DEBUG)) {
        return;
    }

    ucs_string_buffer_appendf(&strb, "[");
    ucs_array_for_each_index(nic, nic_idx, &group->nics) {
        ucs_assert((nic->num_sys_devs > 0) &&
                   (nic->sys_devs[0] != UCS_SYS_DEVICE_ID_UNKNOWN));

        if (nic_idx == 0) {
            ucs_string_buffer_appendf(&strb, " ");
        } else {
            ucs_string_buffer_appendf(&strb, ", ");
        }

        has_owner = 0;
        ucs_array_for_each_index(gpu, gpu_idx, &group->gpus) {
            ucs_assert((gpu->num_sys_devs > 0) &&
                       (gpu->sys_devs[0] != UCS_SYS_DEVICE_ID_UNKNOWN));

            nic_sys_dev_bitmap =
                    ucp_gpu_nic_assignment_lookup(assignment, gpu->sys_devs[0]);
            ucs_assert(nic_sys_dev_bitmap != NULL);

            if (!ucp_gpu_nic_bitmap_get(nic_sys_dev_bitmap, nic->sys_devs[0])) {
                continue;
            }

            ucs_string_buffer_appendf(&strb, "%s%zu", has_owner ? "/" : "",
                                      gpu_idx);
            has_owner = 1;
        }

        if (!has_owner) {
            ucs_string_buffer_appendf(&strb, "X");
        }
    }
    ucs_string_buffer_appendf(&strb, " ]");

    ucs_debug("group #%zu (%zu gpus, %zu nics) assignment: %s", group_idx,
              num_gpus, num_nics, ucs_string_buffer_cstr(&strb));
    ucs_string_buffer_cleanup(&strb);
}

static ucs_status_t
ucp_gpu_nic_assignment_allocate_bitmaps(ucp_gpu_nic_assignment_t *assignment,
                                        const ucs_topo_groups_t *groups)
{
    size_t num_bitmaps = 0;
    const ucs_topo_group_t *group;

    ucs_array_for_each(group, groups) {
        num_bitmaps += ucs_array_length(&group->gpus);
    }

    if (num_bitmaps == 0) {
        return UCS_OK;
    }

    if (num_bitmaps > UCP_GPU_NIC_BITMAP_INDEX_INVALID) {
        ucs_error("gpu-nic sys_dev bitmap count %zu exceeds limit %u",
                  num_bitmaps, (unsigned)UCP_GPU_NIC_BITMAP_INDEX_INVALID);
        return UCS_ERR_EXCEEDS_LIMIT;
    }

    assignment->nic_sys_dev_bitmaps =
            ucs_calloc(num_bitmaps, sizeof(*assignment->nic_sys_dev_bitmaps),
                       "gpu_nic_sys_dev_bitmaps");
    if (assignment->nic_sys_dev_bitmaps == NULL) {
        ucs_error("failed to allocate gpu-nic sys_dev bitmaps");
        return UCS_ERR_NO_MEMORY;
    }

    assignment->num_bitmaps = num_bitmaps;
    return UCS_OK;
}

static void ucp_gpu_nic_assignment_map_gpu(ucp_gpu_nic_assignment_t *assignment,
                                           const ucs_topo_group_element_t *gpu,
                                           size_t nic_sys_dev_bitmap_idx)
{
    const ucs_sys_device_t *gpu_sys_dev;

    ucs_assertv((gpu->num_sys_devs > 0) && (gpu->num_sys_devs <=
                                            UCS_TOPO_MAX_SYS_DEVS_PER_ELEMENT),
                "invalid num_sys_devs: %zu", gpu->num_sys_devs);
    ucs_assertv(nic_sys_dev_bitmap_idx < assignment->num_bitmaps,
                "invalid nic_sys_dev_bitmap_idx: %zu (num_bitmaps: %zu)",
                nic_sys_dev_bitmap_idx, assignment->num_bitmaps);

    /* Assign the same bitmap to all devices under the same GPU. */
    ucs_carray_for_each(gpu_sys_dev, gpu->sys_devs, gpu->num_sys_devs) {
        ucs_assert(*gpu_sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN);
        ucs_assertv(assignment->bitmap_idx_by_gpu_sys_dev[*gpu_sys_dev] ==
                            UCP_GPU_NIC_BITMAP_INDEX_INVALID,
                    "gpu_sys_dev %u is already assigned to bitmap %zu",
                    *gpu_sys_dev, nic_sys_dev_bitmap_idx);

        assignment->bitmap_idx_by_gpu_sys_dev[*gpu_sys_dev] = (uint8_t)
                nic_sys_dev_bitmap_idx;
    }
}

static ucs_status_t
ucp_gpu_nic_assignment_init(ucp_gpu_nic_assignment_t *assignment,
                            const ucs_topo_groups_t *groups)
{
    size_t bitmap_idx = 0;
    const ucs_topo_group_t *group;
    const ucs_topo_group_element_t *gpu;
    ucs_status_t status;

    assignment->nic_sys_dev_bitmaps = NULL;
    assignment->num_bitmaps         = 0;
    memset(assignment->bitmap_idx_by_gpu_sys_dev,
           UCP_GPU_NIC_BITMAP_INDEX_INVALID,
           sizeof(assignment->bitmap_idx_by_gpu_sys_dev));

    status = ucp_gpu_nic_assignment_allocate_bitmaps(assignment, groups);
    if (status != UCS_OK) {
        return status;
    }

    /* Map every GPU to a bitmap */
    ucs_array_for_each(group, groups) {
        ucs_array_for_each(gpu, &group->gpus) {
            ucp_gpu_nic_assignment_map_gpu(assignment, gpu, bitmap_idx++);
        }
    }

    ucs_assertv(bitmap_idx == assignment->num_bitmaps,
                "not all bitmaps are mapped: %zu != %zu", bitmap_idx,
                assignment->num_bitmaps);

    return UCS_OK;
}

static void
ucp_gpu_nic_assignment_add_group(ucp_gpu_nic_assignment_t *assignment,
                                 const ucs_topo_group_t *group,
                                 ucp_gpu_nic_assignment_policy_t policy,
                                 size_t group_idx)
{
    size_t num_gpus = ucs_array_length(&group->gpus);
    ucp_gpu_nic_sys_dev_bitmap_t *nic_sys_dev_bitmap;
    const ucs_topo_group_element_t *nic;
    const ucs_topo_group_element_t *gpu;
    size_t nic_idx;
    size_t gpu_idx;

    ucs_array_for_each_index(nic, nic_idx, &group->nics) {
        gpu_idx = ucp_gpu_nic_assignment_get_gpu_idx(policy, num_gpus, nic_idx);
        gpu     = &ucs_array_elem(&group->gpus, gpu_idx);

        ucs_assert((gpu->num_sys_devs > 0) &&
                   (gpu->sys_devs[0] != UCS_SYS_DEVICE_ID_UNKNOWN));

        nic_sys_dev_bitmap =
                ucp_gpu_nic_assignment_lookup_mut(assignment, gpu->sys_devs[0]);
        ucs_assert(nic_sys_dev_bitmap != NULL);

        ucp_gpu_nic_bitmap_add_nic(nic_sys_dev_bitmap, nic);
    }

    ucp_gpu_nic_assignment_log_group(assignment, group, group_idx);
}

ucs_status_t
ucp_gpu_nic_assignment_build(const ucs_topo_groups_t *groups,
                             ucp_gpu_nic_assignment_policy_t policy,
                             ucp_gpu_nic_assignment_t *assignment_p)
{
    ucp_gpu_nic_assignment_t assignment;
    const ucs_topo_group_t *group;
    ucs_status_t status;
    size_t group_idx;

    ucs_assert(groups != NULL);
    ucs_assertv(policy < UCP_GPU_NIC_ASSIGNMENT_POLICY_LAST,
                "invalid gpu-nic assignment policy %d", (int)policy);

    status = ucp_gpu_nic_assignment_init(&assignment, groups);
    if (status != UCS_OK) {
        return status;
    }

    ucs_array_for_each_index(group, group_idx, groups) {
        if (ucs_array_is_empty(&group->gpus)) {
            ucs_debug("group #%zu has 0 GPUs, skipping", group_idx);
            continue;
        }

        ucp_gpu_nic_assignment_add_group(&assignment, group, policy, group_idx);
    }

    ucs_assert(assignment_p != NULL);
    *assignment_p = assignment;

    return UCS_OK;
}

int ucp_gpu_nic_bitmap_get(const ucp_gpu_nic_sys_dev_bitmap_t *bitmap,
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
    ucs_free(assignment->nic_sys_dev_bitmaps);
}
