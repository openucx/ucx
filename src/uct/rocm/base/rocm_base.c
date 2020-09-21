/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rocm_base.h"

#include <ucs/sys/module.h>

#include <hsa_ext_amd.h>
#include <pthread.h>


#define MAX_AGENTS 16
static struct agents {
    hsa_agent_t agents[MAX_AGENTS];
    int num;
    hsa_agent_t gpu_agents[MAX_AGENTS];
    int num_gpu;
} uct_rocm_base_agents;

int uct_rocm_base_get_gpu_agents(hsa_agent_t **agents)
{
    *agents = uct_rocm_base_agents.gpu_agents;
    return uct_rocm_base_agents.num_gpu;
}

static hsa_status_t uct_rocm_hsa_agent_callback(hsa_agent_t agent, void* data)
{
    hsa_device_type_t device_type;

    ucs_assert(uct_rocm_base_agents.num < MAX_AGENTS);

    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
    if (device_type == HSA_DEVICE_TYPE_CPU) {
        ucs_trace("%d found cpu agent %lu", getpid(), agent.handle);
    }
    else if (device_type == HSA_DEVICE_TYPE_GPU) {
        uint32_t bdfid = 0;
        uct_rocm_base_agents.gpu_agents[uct_rocm_base_agents.num_gpu++] = agent;
        hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &bdfid);
        ucs_trace("%d found gpu agent %lu bdfid %x", getpid(), agent.handle, bdfid);
    }
    else {
        ucs_trace("%d found unknown agent %lu", getpid(), agent.handle);
    }

    uct_rocm_base_agents.agents[uct_rocm_base_agents.num++] = agent;
    return HSA_STATUS_SUCCESS;
}

hsa_status_t uct_rocm_base_init(void)
{
    static pthread_mutex_t rocm_init_mutex = PTHREAD_MUTEX_INITIALIZER;
    static volatile int rocm_ucx_initialized = 0;
    hsa_status_t status;

    if (pthread_mutex_lock(&rocm_init_mutex) == 0) {
        if (rocm_ucx_initialized) {
            status =  HSA_STATUS_SUCCESS;
            goto end;
        }
    } else  {
        ucs_error("Could not take mutex");
        status = HSA_STATUS_ERROR;
        return status;
    }

    memset(&uct_rocm_base_agents, 0, sizeof(uct_rocm_base_agents));

    status = hsa_init();
    if (status != HSA_STATUS_SUCCESS) {
        ucs_debug("Failure to open HSA connection: 0x%x", status);
        goto end;
    }

    status = hsa_iterate_agents(uct_rocm_hsa_agent_callback, NULL);
    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
        ucs_debug("Failure to iterate HSA agents: 0x%x", status);
        goto end;
    }

    rocm_ucx_initialized = 1;

end:
    pthread_mutex_unlock(&rocm_init_mutex);
    return status;
}

ucs_status_t
uct_rocm_base_query_md_resources(uct_component_h component,
                                 uct_md_resource_desc_t **resources_p,
                                 unsigned *num_resources_p)
{
    if (uct_rocm_base_init() != HSA_STATUS_SUCCESS) {
        ucs_debug("could not initialize ROCm support");
        return uct_md_query_empty_md_resource(resources_p, num_resources_p);
    }

    return uct_md_query_single_md_resource(component, resources_p,
                                           num_resources_p);
}

ucs_status_t uct_rocm_base_query_devices(uct_md_h md,
                                         uct_tl_device_resource_t **tl_devices_p,
                                         unsigned *num_tl_devices_p)
{
    return uct_single_device_resource(md, md->component->name,
                                      UCT_DEVICE_TYPE_ACC, tl_devices_p,
                                      num_tl_devices_p);
}

hsa_agent_t uct_rocm_base_get_dev_agent(int dev_num)
{
    ucs_assert(dev_num < uct_rocm_base_agents.num);
    return uct_rocm_base_agents.agents[dev_num];
}

int uct_rocm_base_get_dev_num(hsa_agent_t agent)
{
    int i;

    for (i = 0; i < uct_rocm_base_agents.num; i++) {
        if (uct_rocm_base_agents.agents[i].handle == agent.handle)
            return i;
    }
    ucs_assert(0);
    return -1;
}

int uct_rocm_base_is_gpu_agent(hsa_agent_t agent)
{
    int i;

    for (i = 0; i < uct_rocm_base_agents.num_gpu; i++) {
        if (uct_rocm_base_agents.gpu_agents[i].handle == agent.handle)
            return 1;
    }
    return 0;
}

hsa_status_t uct_rocm_base_get_ptr_info(void *ptr, size_t size,
                                        void **base_ptr, size_t *base_size,
                                        hsa_agent_t *agent)
{
    hsa_status_t status;
    hsa_amd_pointer_info_t info;

    info.size = sizeof(hsa_amd_pointer_info_t);
    status = hsa_amd_pointer_info(ptr, &info, NULL, NULL, NULL);
    if (status != HSA_STATUS_SUCCESS) {
        ucs_error("get pointer info fail %p", ptr);
        return status;
    }

    if (info.type != HSA_EXT_POINTER_TYPE_HSA)
        return HSA_STATUS_ERROR;

    *agent = info.agentOwner;

    if (base_ptr)
        *base_ptr = info.agentBaseAddress;
    if (base_size)
        *base_size = info.sizeInBytes;

    return HSA_STATUS_SUCCESS;
}

ucs_status_t uct_rocm_base_detect_memory_type(uct_md_h md, const void *addr,
                                              size_t length,
                                              ucs_memory_type_t *mem_type_p)
{
    hsa_status_t status;
    hsa_amd_pointer_info_t info;

    *mem_type_p = UCS_MEMORY_TYPE_HOST;
    if (addr == NULL) {
        return UCS_OK;
    }

    info.size = sizeof(hsa_amd_pointer_info_t);
    status = hsa_amd_pointer_info((void*)addr, &info, NULL, NULL, NULL);
    if ((status == HSA_STATUS_SUCCESS) &&
        (info.type == HSA_EXT_POINTER_TYPE_HSA)) {
        hsa_device_type_t dev_type;

        status = hsa_agent_get_info(info.agentOwner, HSA_AGENT_INFO_DEVICE, &dev_type);
        if ((status == HSA_STATUS_SUCCESS) &&
            (dev_type == HSA_DEVICE_TYPE_GPU)) {
            *mem_type_p = UCS_MEMORY_TYPE_ROCM;
            return UCS_OK;
        }
    }

    return UCS_ERR_INVALID_ADDR;
}

ucs_status_t uct_rocm_base_mem_query(uct_md_h md, const void *addr,
                                     const size_t length,
                                     uct_md_mem_attr_t *mem_attr_p)
{
    ucs_status_t status;
    ucs_memory_type_t mem_type;

    status = uct_rocm_base_detect_memory_type(md, addr, length, &mem_type);
    if (status != UCS_OK) {
        return status;
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_FIELD_MEM_TYPE) {
        mem_attr_p->mem_type = mem_type;
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_FIELD_SYS_DEV) {
        mem_attr_p->sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
    }

    return UCS_OK;
}

UCS_MODULE_INIT() {
    UCS_MODULE_FRAMEWORK_DECLARE(uct_rocm);
    UCS_MODULE_FRAMEWORK_LOAD(uct_rocm, 0);
    return UCS_OK;
}
