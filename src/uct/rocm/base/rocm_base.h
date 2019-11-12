/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */


#ifndef ROCM_BASE_H
#define ROCM_BASE_H

#include <uct/base/uct_iface.h>
#include <uct/base/uct_md.h>
#include <hsa.h>


hsa_status_t uct_rocm_base_init(void);
ucs_status_t uct_rocm_base_query_md_resources(uct_component_h component,
                                              uct_md_resource_desc_t **resources_p,
                                              unsigned *num_resources_p);
ucs_status_t uct_rocm_base_query_devices(uct_md_h md,
                                         uct_tl_device_resource_t **tl_devices_p,
                                         unsigned *num_tl_devices_p);
hsa_agent_t uct_rocm_base_get_dev_agent(int dev_num);
int uct_rocm_base_is_gpu_agent(hsa_agent_t agent);
int uct_rocm_base_get_gpu_agents(hsa_agent_t **agents);
int uct_rocm_base_get_dev_num(hsa_agent_t agent);
hsa_status_t uct_rocm_base_get_ptr_info(void *ptr, size_t size,
                                        void **base_ptr, size_t *base_size,
                                        hsa_agent_t *agent);
ucs_status_t uct_rocm_base_detect_memory_type(uct_md_h md, const void *addr,
                                              size_t length,
                                              ucs_memory_type_t *mem_type_p);

#endif
