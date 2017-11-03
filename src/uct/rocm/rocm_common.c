/*
 * Copyright (C) Advanced Micro Devices, Inc. 2016 - 2017. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "rocm_common.h"
#include <pthread.h>

#include <uct/base/uct_log.h>



/** Mutex to guarantee that initialization will be atomic */
static pthread_mutex_t rocm_init_mutex = PTHREAD_MUTEX_INITIALIZER;


/** Max number of HSA agents supported */
#define MAX_HSA_AGENTS      64


/** Structure to keep all collected configuration info */
static struct {
    struct {
        struct {
            uint32_t                bus;    /**< PCI Bus id */
            uint32_t                device; /**< PCI Device id */
            uint32_t                func;   /**< PCI Function id */
            hsa_amd_memory_pool_t   pool;   /**< Global pool associated with agent.
                                              @note Current we assume that there
                                              is only one global pool per agent
                                              base on the current behaviour */
        } gpu_info[MAX_HSA_AGENTS];
        hsa_agent_t gpu_agent[MAX_HSA_AGENTS];/**< HSA GPU Agent handles */
        struct {
            hsa_agent_t             agent;  /**< HSA Agent handle for CPU */
            hsa_amd_memory_pool_t   pool;   /**< Global pool associated with agent.
                                             @note Current we assume that there
                                             is only one global pool per agent
                                             base on the current behaviour */
        } cpu;
    } agents;
    int     num_of_gpu;
} uct_rocm_cfg;


/** Internal structure to store information about memory */
typedef struct {
    void                   *ptr;
    hsa_amd_pointer_info_t  info;
    uint32_t                num_agents_accessible;
    hsa_agent_t             accessible[MAX_HSA_AGENTS];
} uct_rocm_ptr_t;


/** Callback to enumerate pools for given agent.
 *  Try to find global pool assuming one global pool per agent.
*/
static hsa_status_t uct_rocm_hsa_amd_memory_pool_callback(
                                                        hsa_amd_memory_pool_t memory_pool,
                                                        void* data)
{
    hsa_status_t status;
    hsa_amd_segment_t amd_segment;

    status = hsa_amd_memory_pool_get_info(memory_pool,
                                          HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                          &amd_segment);

    if (status != HSA_STATUS_SUCCESS) {
        ucs_error("Failure to get pool info: 0x%x", status);
        return status;
    }

    if (amd_segment ==  HSA_AMD_SEGMENT_GLOBAL) {
        *(hsa_amd_memory_pool_t *)data = memory_pool;
        ucs_debug("Found global pool: 0x%lx", memory_pool.handle);
        return HSA_STATUS_INFO_BREAK;
    }

    return HSA_STATUS_SUCCESS;
}

/** Callback to enumerate HSA agents */
static hsa_status_t uct_rocm_hsa_agent_callback(hsa_agent_t agent, void* data)
{
    uint32_t bdfid;
    hsa_device_type_t device_type;
    hsa_status_t status;

    ucs_debug("hsa_agent_callback: Agent  0x%lx", agent.handle);

    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);

    if (status != HSA_STATUS_SUCCESS) {
        ucs_error("Failure to get device type: 0x%x", status);
        return status;
    }

    if (device_type == HSA_DEVICE_TYPE_GPU) {

        status = hsa_agent_get_info(agent, HSA_AMD_AGENT_INFO_BDFID, &bdfid);

        if (status != HSA_STATUS_SUCCESS) {
            ucs_warn("Failure to get pci info: 0x%x", status);
            return status;
        }

        uct_rocm_cfg.agents.gpu_agent[uct_rocm_cfg.num_of_gpu] = agent;
        uct_rocm_cfg.agents.gpu_info[uct_rocm_cfg.num_of_gpu].bus = (bdfid >> 8) & 0xff;
        uct_rocm_cfg.agents.gpu_info[uct_rocm_cfg.num_of_gpu].device = (bdfid >> 3) & 0x1F;
        uct_rocm_cfg.agents.gpu_info[uct_rocm_cfg.num_of_gpu].func = bdfid & 0x7;

        ucs_debug("Found GPU agent : 0x%lx. [ B#%02d, D#%02d, F#%02d ]",
                  uct_rocm_cfg.agents.gpu_agent[uct_rocm_cfg.num_of_gpu].handle,
                  uct_rocm_cfg.agents.gpu_info[uct_rocm_cfg.num_of_gpu].bus,
                  uct_rocm_cfg.agents.gpu_info[uct_rocm_cfg.num_of_gpu].device,
                  uct_rocm_cfg.agents.gpu_info[uct_rocm_cfg.num_of_gpu].func);


        uct_rocm_cfg.agents.gpu_info[uct_rocm_cfg.num_of_gpu].pool.handle
                                                            = (uint64_t) -1;
        status = hsa_amd_agent_iterate_memory_pools(agent,
                                                    uct_rocm_hsa_amd_memory_pool_callback,
                                                    &uct_rocm_cfg.agents.gpu_info[uct_rocm_cfg.num_of_gpu].pool);

        if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
            ucs_error("Failure to iterate regions: 0x%x\n", status);
            return status;
        }

        if (uct_rocm_cfg.agents.gpu_info[uct_rocm_cfg.num_of_gpu].pool.handle
                                                    == (uint64_t)-1) {
            ucs_warn("Could not find memory pool for GPU agent");
        }

        uct_rocm_cfg.num_of_gpu++;

    } else  if (device_type == HSA_DEVICE_TYPE_CPU) {
        uct_rocm_cfg.agents.cpu.agent = agent;
        ucs_debug("Found CPU agent : 0x%lx", uct_rocm_cfg.agents.cpu.agent.handle);

        uct_rocm_cfg.agents.cpu.pool.handle = (uint64_t) -1;
        status = hsa_amd_agent_iterate_memory_pools(agent,
                                                    uct_rocm_hsa_amd_memory_pool_callback,
                                                    &uct_rocm_cfg.agents.cpu.pool);

        if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
            ucs_error("Failure to iterate memory pools: 0x%x", status);
            return status;
        }

        if (uct_rocm_cfg.agents.cpu.pool.handle == (uint64_t)-1) {
            ucs_warn("Could not find memory pool for CPU agent");
        }
    }

    /* Keep iterating */
    return HSA_STATUS_SUCCESS;
}


hsa_status_t uct_rocm_init()
{
    /** Flag to specify if ROCm UCX support was initialized or not */
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

    /* Initialize HSA RT just in case if it was not initialized before */
    status = hsa_init();

    if (status != HSA_STATUS_SUCCESS) {
        ucs_error("Failure to open HSA connection: 0x%x", status);
        goto end;
    }

    /* Collect information about GPU agents */
    status = hsa_iterate_agents(uct_rocm_hsa_agent_callback, NULL);

    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
        ucs_error("Failure to iterate HSA agents: 0x%x", status);
        goto end;
    }

    rocm_ucx_initialized = 1;

end:
    pthread_mutex_unlock(&rocm_init_mutex);
    return status;
}

int uct_rocm_is_ptr_gpu_accessible(void *ptr, void **gpu_ptr)
{
    hsa_amd_pointer_info_t info;
    info.size = sizeof(hsa_amd_pointer_info_t);

    hsa_status_t status = hsa_amd_pointer_info(ptr, (hsa_amd_pointer_info_t *)&info,
                                               NULL, NULL, NULL);

    if (status == HSA_STATUS_SUCCESS) {
        if (info.type != HSA_EXT_POINTER_TYPE_UNKNOWN) {
            if (gpu_ptr) {
                *gpu_ptr = info.agentBaseAddress;

                /** Note: hsa_amd_pointer_info() will return information
                    about base address of allocated pool or registered memory.
                    Accordingly if passed address is "inside" of range 
                    we need to correspondingly adjust returned information */
                if (info.type == HSA_EXT_POINTER_TYPE_LOCKED) {
                    /* This is memory allocated outside of ROCm stack */
                    *gpu_ptr += ptr - info.hostBaseAddress;
                }
                else if (info.type == HSA_EXT_POINTER_TYPE_HSA) {
                    /* This is the GPU pointer */
                    *gpu_ptr += ptr - info.agentBaseAddress;
                }
                else {
                    /* Assume that "ptr" is GPU pointer */
                    *gpu_ptr += ptr - info.agentBaseAddress;
                }
            }

            ucs_trace("%p is GPU accessible (agent addr %p, Host Base %p)",
                      ptr, info.agentBaseAddress, info.hostBaseAddress);
            return 1;
        }
    }

    ucs_trace_func("%p is not GPU accessible", ptr);
    return 0;
}


hsa_status_t uct_rocm_memory_lock(void *ptr, size_t size, void **gpu_ptr)
{
    /* We need to lock / register memory on all GPUs because we do not know
       the location of other memory */
    hsa_status_t status = hsa_amd_memory_lock(ptr, size,
                                              uct_rocm_cfg.agents.gpu_agent,
                                              uct_rocm_cfg.num_of_gpu,
                                              gpu_ptr);

    if (status != HSA_STATUS_SUCCESS) {
        ucs_error("Failed to lock memory (%p): 0x%x\n", ptr, status);
    }

    return status;
}
