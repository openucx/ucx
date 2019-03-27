/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */


#ifndef ROCM_BASE_H
#define ROCM_BASE_H

#include <uct/base/uct_md.h>
#include <hsa.h>

hsa_status_t uct_rocm_base_init(void);
hsa_agent_t uct_rocm_base_get_dev_agent(int dev_num);
int uct_rocm_base_is_gpu_agent(hsa_agent_t agent);
int uct_rocm_base_get_gpu_agents(hsa_agent_t **agents);
int uct_rocm_base_get_dev_num(hsa_agent_t agent);
hsa_status_t uct_rocm_base_get_ptr_info(void *ptr, size_t size,
                                        void **base_ptr, size_t *base_size,
                                        hsa_agent_t *agent);
ucs_status_t uct_rocm_base_detect_memory_type(uct_md_h md, void *addr, size_t length,
                                              uct_memory_type_t *mem_type);

#endif
