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
#include <ucs/sys/compiler.h>

#include <string.h>


#define UCP_GPU_NIC_STRING_BUFFERS_ONSTACK(_var, _capacity) \
    UCS_STRING_BUFFER_ONSTACK(UCS_PP_TOKENPASTE(_var, _names), (_capacity)); \
    UCS_STRING_BUFFER_ONSTACK(UCS_PP_TOKENPASTE(_var, _sys_devs), (_capacity)); \
    ucp_nics_string_buffers_t _var = {&UCS_PP_TOKENPASTE(_var, _names), \
                                      &UCS_PP_TOKENPASTE(_var, _sys_devs), 0}


typedef struct {
    ucs_string_buffer_t *names;
    ucs_string_buffer_t *sys_devs;
    size_t              count;
} ucp_nics_string_buffers_t;


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

    ucs_assertv(nic_sys_dev_bitmap_idx < assignment->num_bitmaps,
                "invalid nic_sys_dev_bitmap_idx %u (num_bitmaps: %zu) assigned "
                "to gpu_sys_dev %u",
                nic_sys_dev_bitmap_idx, assignment->num_bitmaps, gpu_sys_dev);

    return &assignment->nic_sys_dev_bitmaps[nic_sys_dev_bitmap_idx];
}

static void ucp_gpu_nic_assignment_append_device_names(
        const ucs_topo_group_element_t *element, ucs_string_buffer_t *strb)
{
    size_t i;

    ucs_assertv((element->num_sys_devs > 0) &&
                        (element->num_sys_devs <=
                         UCS_TOPO_MAX_SYS_DEVS_PER_ELEMENT),
                "invalid num_sys_devs: %zu", element->num_sys_devs);

    for (i = 0; i < element->num_sys_devs; ++i) {
        ucs_string_buffer_appendf(strb, "%s/",
                                  ucs_topo_sys_device_get_name(
                                          element->sys_devs[i]));
    }
    ucs_string_buffer_rtrim(strb, "/");
}

static void
ucp_gpu_nic_assignment_append_nic(const ucs_topo_group_element_t *nic,
                                  ucp_nics_string_buffers_t *strbs)
{
    if (strbs->count > 0) {
        ucs_string_buffer_appendf(strbs->names, " ");
        ucs_string_buffer_appendf(strbs->sys_devs, " ");
    }

    ++strbs->count;

    ucp_gpu_nic_assignment_append_device_names(nic, strbs->names);
    ucs_string_buffer_append_array(strbs->sys_devs, "/", "%hhu", nic->sys_devs,
                                   nic->num_sys_devs);
}

static void
ucp_gpu_nic_assignment_log_gpu(const ucp_gpu_nic_assignment_t *assignment,
                               const ucs_topo_group_t *group,
                               const ucs_topo_group_element_t *gpu,
                               ucp_gpu_nic_sys_dev_bitmap_t *assigned_nics)
{
    UCS_STRING_BUFFER_ONSTACK(gpu_strb, 128);
    UCP_GPU_NIC_STRING_BUFFERS_ONSTACK(nic_strbs, 256);
    char bdf_name[UCS_SYS_BDF_NAME_MAX];
    const ucp_gpu_nic_sys_dev_bitmap_t *nic_sys_dev_bitmap;
    const ucs_topo_group_element_t *nic;

    ucs_assert((gpu->num_sys_devs > 0) &&
               (gpu->sys_devs[0] != UCS_SYS_DEVICE_ID_UNKNOWN));

    nic_sys_dev_bitmap = ucp_gpu_nic_assignment_lookup(assignment,
                                                       gpu->sys_devs[0]);
    ucs_assert(nic_sys_dev_bitmap != NULL);
    UCS_STATIC_BITMAP_OR_INPLACE(assigned_nics, *nic_sys_dev_bitmap);

    ucp_gpu_nic_assignment_append_device_names(gpu, &gpu_strb);

    ucs_string_buffer_appendf(&gpu_strb, " (bdf %s sys_dev ",
                              ucs_topo_sys_device_bdf_name(gpu->sys_devs[0],
                                                           bdf_name,
                                                           sizeof(bdf_name)));
    ucs_string_buffer_append_array(&gpu_strb, "/", "%hhu", gpu->sys_devs,
                                   gpu->num_sys_devs);
    ucs_string_buffer_appendf(&gpu_strb, ")");

    ucs_array_for_each(nic, &group->nics) {
        if (!ucp_gpu_nic_bitmap_get(nic_sys_dev_bitmap, nic->sys_devs[0])) {
            continue;
        }

        ucp_gpu_nic_assignment_append_nic(nic, &nic_strbs);
    }

    if (nic_strbs.count == 0) {
        ucs_debug("gpu %s is assigned 0 nics",
                  ucs_string_buffer_cstr(&gpu_strb));
    } else {
        ucs_debug("gpu %s is assigned %zu nics: [%s] sys_devs [%s]",
                  ucs_string_buffer_cstr(&gpu_strb), nic_strbs.count,
                  ucs_string_buffer_cstr(nic_strbs.names),
                  ucs_string_buffer_cstr(nic_strbs.sys_devs));
    }
}

static void ucp_gpu_nic_assignment_log_unassigned_nics(
        const ucs_topo_groups_t *groups,
        const ucp_gpu_nic_sys_dev_bitmap_t *assigned_nics)
{
    UCP_GPU_NIC_STRING_BUFFERS_ONSTACK(nic_strbs, 256);
    const ucs_topo_group_element_t *nic;
    const ucs_topo_group_t *nic_group;

    ucs_array_for_each(nic_group, groups) {
        ucs_array_for_each(nic, &nic_group->nics) {
            if (ucp_gpu_nic_bitmap_get(assigned_nics, nic->sys_devs[0])) {
                continue;
            }

            ucp_gpu_nic_assignment_append_nic(nic, &nic_strbs);
        }
    }

    if (nic_strbs.count == 0) {
        ucs_debug("all nics are assigned");
    } else {
        ucs_debug("%zu nics are unassigned: [%s] sys_devs [%s]",
                  nic_strbs.count, ucs_string_buffer_cstr(nic_strbs.names),
                  ucs_string_buffer_cstr(nic_strbs.sys_devs));
    }
}

static void
ucp_gpu_nic_assignment_log(const ucp_gpu_nic_assignment_t *assignment,
                           const ucs_topo_groups_t *groups)
{
    ucp_gpu_nic_sys_dev_bitmap_t assigned_nics =
            UCS_STATIC_BITMAP_ZERO_INITIALIZER;
    const ucs_topo_group_element_t *gpu;
    const ucs_topo_group_t *group;

    if (!ucs_log_is_enabled(UCS_LOG_LEVEL_DEBUG)) {
        return;
    }

    ucs_array_for_each(group, groups) {
        ucs_array_for_each(gpu, &group->gpus) {
            ucp_gpu_nic_assignment_log_gpu(assignment, group, gpu,
                                           &assigned_nics);
        }
    }

    ucp_gpu_nic_assignment_log_unassigned_nics(groups, &assigned_nics);
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
                    "gpu_sys_dev %u is already assigned to bitmap %u",
                    *gpu_sys_dev,
                    assignment->bitmap_idx_by_gpu_sys_dev[*gpu_sys_dev]);

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
                                 ucp_gpu_nic_assignment_policy_t policy)
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

        nic_sys_dev_bitmap = ucs_const_cast(
                ucp_gpu_nic_sys_dev_bitmap_t*,
                ucp_gpu_nic_assignment_lookup(assignment, gpu->sys_devs[0]));
        ucs_assert(nic_sys_dev_bitmap != NULL);

        ucp_gpu_nic_bitmap_add_nic(nic_sys_dev_bitmap, nic);
    }
}

ucs_status_t
ucp_gpu_nic_assignment_build(const ucs_topo_groups_t *groups,
                             ucp_gpu_nic_assignment_policy_t policy,
                             ucp_gpu_nic_assignment_t *assignment_p)
{
    ucp_gpu_nic_assignment_t assignment;
    const ucs_topo_group_t *group;
    ucs_status_t status;

    ucs_assert(groups != NULL);
    ucs_assertv(policy < UCP_GPU_NIC_ASSIGNMENT_POLICY_LAST,
                "invalid gpu-nic assignment policy %d", (int)policy);

    status = ucp_gpu_nic_assignment_init(&assignment, groups);
    if (status != UCS_OK) {
        return status;
    }

    ucs_array_for_each(group, groups) {
        if (ucs_array_is_empty(&group->gpus)) {
            continue;
        }

        ucp_gpu_nic_assignment_add_group(&assignment, group, policy);
    }

    ucp_gpu_nic_assignment_log(&assignment, groups);

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
    ucs_assert(assignment != NULL);
    ucs_free(assignment->nic_sys_dev_bitmaps);
    assignment->nic_sys_dev_bitmaps = NULL;
    assignment->num_bitmaps         = 0;
    memset(assignment->bitmap_idx_by_gpu_sys_dev,
           UCP_GPU_NIC_BITMAP_INDEX_INVALID,
           sizeof(assignment->bitmap_idx_by_gpu_sys_dev));
}
