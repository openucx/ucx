/*
 * Copyright (C) Advanced Micro Devices, Inc. 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_MUSA_BASE_H
#define UCT_MUSA_BASE_H

#include <uct/base/uct_iface.h>
#include <uct/base/uct_md.h>
#include <uct/musa/musa_utils.h>

#include <musa.h>
#include <musa_runtime_api.h>

typedef enum {
    UCT_MUSA_POINTER_HOST,
    UCT_MUSA_POINTER_DEVICE,
    UCT_MUSA_POINTER_MANAGED
} uct_musa_pointer_kind_t;

typedef struct {
    uct_musa_pointer_kind_t kind;
    int                     device;
    void                   *base_address;
    size_t                  alloc_length;
} uct_musa_pointer_info_t;

ucs_status_t
uct_musa_base_query_md_resources(uct_component_h component,
                                 uct_md_resource_desc_t **resources_p,
                                 unsigned *num_resources_p);

ucs_status_t
uct_musa_base_query_devices(uct_md_h md,
                            uct_tl_device_resource_t **tl_devices_p,
                            unsigned *num_tl_devices_p);

ucs_status_t uct_musa_base_get_device_count(unsigned *count_p);

ucs_status_t uct_musa_base_query_pointer(const void *address, size_t length,
                                         uct_musa_pointer_info_t *info);

ucs_status_t uct_musa_base_mem_query(const void *address, size_t length,
                                     int have_dmabuf,
                                     uct_md_mem_attr_v2_t *mem_attr_p);

ucs_status_t uct_musa_base_detect_memory_type(uct_md_h md, const void *address,
                                              size_t length,
                                              ucs_memory_type_t *mem_type_p);

ucs_status_t uct_musa_base_copy(const void *src, void *dst, size_t length,
                                int device);

int uct_musa_base_has_dmabuf(void);

ucs_status_t uct_musa_base_export_dmabuf(const void *address, size_t length,
                                         int *fd_p, size_t *offset_p);

#endif
