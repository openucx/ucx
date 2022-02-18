/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_MD_H
#define UCT_CUDA_MD_H

#include <uct/base/uct_md.h>

#define UCT_CUDA_DEV_NAME_MAX_LEN 64
#define UCT_CUDA_MAX_DEVICES      16


typedef struct uct_cuda_base_sys_dev_map {
    pid_t pid;
    uint8_t count;
    ucs_sys_device_t sys_dev[UCT_CUDA_MAX_DEVICES];
    uint8_t bus_id[UCT_CUDA_MAX_DEVICES];
} uct_cuda_base_sys_dev_map_t;


extern uct_cuda_base_sys_dev_map_t uct_cuda_sys_dev_bus_id_map;


ucs_status_t uct_cuda_base_detect_memory_type(uct_md_h md, const void *address,
                                              size_t length,
                                              ucs_memory_type_t *mem_type_p);

ucs_status_t uct_cuda_base_mem_query(uct_md_h md, const void *address,
                                     size_t length, uct_md_mem_attr_t *mem_attr);

ucs_status_t
uct_cuda_base_query_md_resources(uct_component_t *component,
                                 uct_md_resource_desc_t **resources_p,
                                 unsigned *num_resources_p);

#endif
