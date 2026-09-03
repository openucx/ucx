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

    ucs_assert(policy == UCP_GPU_NIC_ASSIGNMENT_POLICY_FLIP);
    leg = nic_idx / num_gpus;
    if ((leg % 2) != 0) {
        gpu_idx = num_gpus - 1 - gpu_idx;
    }

    return gpu_idx;
}

static void
ucp_gpu_nic_bitmap_add_nic(ucp_gpu_nic_sys_dev_bitmap_t *nic_sys_dev_bitmap,
                           const ucs_topo_nic_t *nic)
{
    const ucs_sys_device_t *port;

    ucs_assert((nic->num_ports > 0) &&
               (nic->num_ports <= UCS_TOPO_MAX_PORTS_PER_NIC));

    ucs_carray_for_each(port, nic->ports, nic->num_ports) {
        ucs_assert(*port != UCS_SYS_DEVICE_ID_UNKNOWN);
        UCS_STATIC_BITMAP_SET(nic_sys_dev_bitmap, *port);
    }
}

/* Log the assignment for a group.
 * For example:
 *  `group #0 (2 gpus, 4 nics) assignment: [ 0, 1, 1, 0 ]`
 *  `group #1 (2 gpus, 4 nics) assignment: [ 0/1, X, X, 0/1 ]`
 */
static void
ucp_gpu_nic_assignment_log_group(const ucp_gpu_nic_assignment_t *assignment,
                                 const ucs_topo_group_t *group,
                                 size_t base_bitmap_idx, size_t group_idx)
{
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    size_t num_gpus          = ucs_array_length(&group->gpus);
    size_t num_nics          = ucs_array_length(&group->nics);
    const ucp_gpu_nic_sys_dev_bitmap_t *nic_sys_dev_bitmap;
    const ucs_topo_nic_t *nic;
    size_t nic_idx;
    size_t gpu_idx;
    int has_owner;

    if (!ucs_log_is_enabled(UCS_LOG_LEVEL_DEBUG)) {
        return;
    }

    ucs_string_buffer_appendf(&strb, "[");
    for (nic_idx = 0; nic_idx < num_nics; ++nic_idx) {
        nic = &ucs_array_elem(&group->nics, nic_idx);

        if (nic_idx == 0) {
            ucs_string_buffer_appendf(&strb, " ");
        } else {
            ucs_string_buffer_appendf(&strb, ", ");
        }

        has_owner = 0;
        for (gpu_idx = 0; gpu_idx < num_gpus; ++gpu_idx) {
            nic_sys_dev_bitmap =
                    &assignment->nic_sys_dev_bitmaps[base_bitmap_idx + gpu_idx];
            if (!ucp_gpu_nic_bitmap_test(nic_sys_dev_bitmap, nic->ports[0])) {
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

static void ucp_gpu_nic_assignment_init(ucp_gpu_nic_assignment_t *assignment)
{
    assignment->nic_sys_dev_bitmaps = NULL;
    assignment->num_bitmaps         = 0;
    memset(assignment->bitmap_idx_by_gpu_sys_dev,
           UCP_GPU_NIC_BITMAP_INDEX_INVALID,
           sizeof(assignment->bitmap_idx_by_gpu_sys_dev));
}

static ucs_status_t
ucp_gpu_nic_assignment_allocate_bitmaps(ucp_gpu_nic_assignment_t *assignment,
                                        const ucs_topo_groups_t *groups)
{
    size_t num_bitmaps = 0;
    const ucs_topo_group_t *group;

    ucs_array_for_each(group, &groups->groups) {
        num_bitmaps += ucs_array_length(&group->gpus);
    }

    if (num_bitmaps == 0) {
        return UCS_OK;
    }

    if (num_bitmaps > UCP_GPU_NIC_BITMAP_INDEX_INVALID) {
        ucs_error("gpu-nic sys-dev bitmap count exceeds limit %u",
                  (unsigned)UCP_GPU_NIC_BITMAP_INDEX_INVALID);
        return UCS_ERR_EXCEEDS_LIMIT;
    }

    assignment->nic_sys_dev_bitmaps =
            ucs_calloc(num_bitmaps, sizeof(*assignment->nic_sys_dev_bitmaps),
                       "gpu_nic_sys_dev_bitmaps");
    if (assignment->nic_sys_dev_bitmaps == NULL) {
        ucs_error("failed to allocate gpu-nic sys-dev bitmaps");
        return UCS_ERR_NO_MEMORY;
    }

    assignment->num_bitmaps = num_bitmaps;
    return UCS_OK;
}

static void ucp_gpu_nic_assignment_map_gpu(ucp_gpu_nic_assignment_t *assignment,
                                           const ucs_topo_gpu_t *gpu,
                                           size_t nic_sys_dev_bitmap_idx)
{
    const ucs_sys_device_t *gpu_sys_dev;

    ucs_assert(nic_sys_dev_bitmap_idx < UCP_GPU_NIC_BITMAP_INDEX_INVALID);

    /* Assign the same bitmap to all devices under the same GPU. */
    ucs_carray_for_each(gpu_sys_dev, gpu->devices, gpu->num_devices) {
        ucs_assert(*gpu_sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN);
        ucs_assert(assignment->bitmap_idx_by_gpu_sys_dev[*gpu_sys_dev] ==
                   UCP_GPU_NIC_BITMAP_INDEX_INVALID);

        assignment->bitmap_idx_by_gpu_sys_dev[*gpu_sys_dev] = (uint8_t)
                nic_sys_dev_bitmap_idx;
    }
}

static void
ucp_gpu_nic_assignment_add_group(ucp_gpu_nic_assignment_t *assignment,
                                 const ucs_topo_group_t *group,
                                 ucp_gpu_nic_assignment_policy_t policy,
                                 size_t base_bitmap_idx, size_t group_idx)
{
    size_t num_gpus = ucs_array_length(&group->gpus);
    size_t nic_idx  = 0;
    ucp_gpu_nic_sys_dev_bitmap_t *nic_sys_dev_bitmap;
    const ucs_topo_nic_t *nic;
    const ucs_topo_gpu_t *gpu;
    size_t gpu_idx;
    size_t bitmap_idx;

    ucs_assert((base_bitmap_idx + num_gpus) <= assignment->num_bitmaps);

    ucs_array_for_each(nic, &group->nics) {
        gpu_idx = ucp_gpu_nic_assignment_get_gpu_idx(policy, num_gpus,
                                                     nic_idx++);
        nic_sys_dev_bitmap =
                &assignment->nic_sys_dev_bitmaps[base_bitmap_idx + gpu_idx];
        ucp_gpu_nic_bitmap_add_nic(nic_sys_dev_bitmap, nic);
    }

    for (gpu_idx = 0; gpu_idx < num_gpus; ++gpu_idx) {
        gpu = &ucs_array_elem(&group->gpus, gpu_idx);

        bitmap_idx         = base_bitmap_idx + gpu_idx;
        nic_sys_dev_bitmap = &assignment->nic_sys_dev_bitmaps[bitmap_idx];
        ucs_assert((gpu->num_devices > 0) &&
                   (gpu->num_devices <= UCS_TOPO_MAX_DEVICES_PER_GPU));

        if (UCS_STATIC_BITMAP_IS_ZERO(*nic_sys_dev_bitmap)) {
            ucs_debug("gpu #%zu in group #%zu has no assigned nics", gpu_idx,
                      group_idx);
        }

        ucp_gpu_nic_assignment_map_gpu(assignment, gpu, bitmap_idx);
    }

    ucp_gpu_nic_assignment_log_group(assignment, group, base_bitmap_idx,
                                     group_idx);
}

ucs_status_t
ucp_gpu_nic_assignment_build(const ucs_topo_groups_t *groups,
                             ucp_gpu_nic_assignment_policy_t policy,
                             ucp_gpu_nic_assignment_t *assignment_p)
{
    size_t base_bitmap_idx = 0;
    ucp_gpu_nic_assignment_t assignment;
    const ucs_topo_group_t *group;
    ucs_status_t status;
    size_t group_idx;

    ucs_assert(groups != NULL);
    ucs_assertv(policy < UCP_GPU_NIC_ASSIGNMENT_POLICY_LAST,
                "invalid gpu-nic assignment policy %d", (int)policy);

    ucp_gpu_nic_assignment_init(&assignment);
    status = ucp_gpu_nic_assignment_allocate_bitmaps(&assignment, groups);
    if (status != UCS_OK) {
        goto err_cleanup_sys_dev_bitmaps;
    }

    for (group_idx = 0; group_idx < ucs_array_length(&groups->groups);
         ++group_idx) {
        group = &ucs_array_elem(&groups->groups, group_idx);

        if (ucs_array_is_empty(&group->gpus)) {
            ucs_debug("group #%zu has 0 GPUs, skipping", group_idx);
            continue;
        }

        ucp_gpu_nic_assignment_add_group(&assignment, group, policy,
                                         base_bitmap_idx, group_idx);
        base_bitmap_idx += ucs_array_length(&group->gpus);
    }

    ucs_assert(base_bitmap_idx == assignment.num_bitmaps);
    ucs_assert(assignment_p != NULL);
    *assignment_p = assignment;

    return UCS_OK;

err_cleanup_sys_dev_bitmaps:
    ucs_free(assignment.nic_sys_dev_bitmaps);
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

    ucs_assert(nic_sys_dev_bitmap_idx < assignment->num_bitmaps);
    return &assignment->nic_sys_dev_bitmaps[nic_sys_dev_bitmap_idx];
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
    ucs_free(assignment->nic_sys_dev_bitmaps);
}
