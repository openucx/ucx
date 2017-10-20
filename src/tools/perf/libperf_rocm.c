/**
* Copyright (C) Advanced Micro Devices, Inc. 2016 - 2017. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#if HAVE_ROCM

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "libperf_int.h"
#include "libperf_rocm.h"

#include <ucs/debug/log.h>
#include <malloc.h>
#include <unistd.h>

#include <hsa.h>
#include <hsa_ext_amd.h>

static hsa_agent_t              hsa_agent;
static hsa_amd_memory_pool_t    hsa_pool;

/* Max. number of HSA agents supported */
#define MAX_HSA_AGENTS          256
static hsa_agent_t              gpu_agents[MAX_HSA_AGENTS];
static int16_t                  num_of_gpu;

static unsigned long hsa_iterate_index;


static hsa_status_t hsa_agent_callback(hsa_agent_t agent, void* data)
{
    char name[64];
    uint32_t bdfid;
    hsa_device_type_t device_type;
    hsa_status_t status;
    unsigned long hsa_agent_index = *(unsigned long *)data;

    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);

    if (status != HSA_STATUS_SUCCESS) {
        ucs_error("Failure to get agent name : 0x%x\n", status);
        return status;
    }

    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);

    if (status != HSA_STATUS_SUCCESS) {
        ucs_error("failure to get device type: 0x%x\n", status);
        return status;
    }

    if (device_type == HSA_DEVICE_TYPE_GPU) {

        status = hsa_agent_get_info(agent, HSA_AMD_AGENT_INFO_BDFID, &bdfid);
        if (status != HSA_STATUS_SUCCESS) {
            ucs_error("failure to get pci info: 0x%x\n", status);
            return status;
        }

        /* bfd: eight-bit pci bus, five-bit device, and three-bit
           function number
        */
        uint32_t Bus	= (bdfid >> 8) & 0xff;
        uint32_t Device	= (bdfid >> 3) & 0x1F;
        uint32_t Func	= bdfid & 0x7;

        ucs_info("Found HSA GPU agent : %s (PCI [ B#%02d, D#%02d, F#%02d ]\n",
                 name, Bus, Device, Func);

        gpu_agents[num_of_gpu] = agent;
        num_of_gpu++;
    }
    else {
        ucs_info("Found HSA CPU agent : %s.\n", name);
    }

    if (hsa_iterate_index == hsa_agent_index) {
        hsa_agent = agent;
    }

    hsa_iterate_index++;
    /* Keep iterating */
    return HSA_STATUS_SUCCESS;
}

static hsa_status_t memory_pool_callback(hsa_amd_memory_pool_t memory_pool, void* data)
{
    hsa_status_t status;
    hsa_amd_memory_pool_global_flag_t global_flags;
    size_t	pool_size;
    hsa_amd_segment_t amd_segment;

    unsigned long hsa_pool_index = *(unsigned long *)data;

    status = hsa_amd_memory_pool_get_info(memory_pool,
                                          HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                          &amd_segment);

    if (status != HSA_STATUS_SUCCESS) {
        ucs_error("Failure to get pool info: 0x%x\n", status);
        return status;
    }

    if (amd_segment ==  HSA_AMD_SEGMENT_GLOBAL) {
        if (hsa_iterate_index == hsa_pool_index) {

            status = hsa_amd_memory_pool_get_info(memory_pool,
                                                  HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                                  &global_flags);

            if (status != HSA_STATUS_SUCCESS) {
                ucs_error("Failure to query global flags: 0x%x\n", status);
                return status;
            }

            status = hsa_amd_memory_pool_get_info(memory_pool,
                                                  HSA_AMD_MEMORY_POOL_INFO_SIZE,
                                                  &pool_size);

            if (status != HSA_STATUS_SUCCESS) {
                ucs_error("Failure to query pool size: 0x%x\n", status);
                return status;
            }

            ucs_info("Found HSA global pool. Flags  : 0x%x, Size  : 0x%lx (%ldMiB)\n",
                     global_flags, pool_size, pool_size / (1024L * 1024L));

            hsa_pool = memory_pool;
            return HSA_STATUS_INFO_BREAK;
        }

        hsa_iterate_index++;
    }

    return HSA_STATUS_SUCCESS;
}

ucs_status_t rocm_init(ucx_perf_params_t *params)
{
    hsa_status_t status;

    ucs_info("Initializing HSA.....\n");

    status = hsa_init();

    if (status != HSA_STATUS_SUCCESS) {
        ucs_error("Failure to open HSA connection: 0x%x\n", status);
        return UCS_ERR_NO_RESOURCE;
    }

    ucs_debug("Searching for HSA agent with index %lu\n", params->hsa_agent_index);

    hsa_iterate_index = 0;
    hsa_agent.handle  = (uint64_t) -1;
    status = hsa_iterate_agents(hsa_agent_callback, &params->hsa_agent_index);

    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
        ucs_error("Failure to iterate HSA agents: 0x%x\n", status);
        return UCS_ERR_NO_RESOURCE;
    }

    if (hsa_agent.handle == (uint64_t)-1) {
        ucs_error("Could not find HSA agent with given index.\n");
        return UCS_ERR_NO_RESOURCE;
    }

    ucs_debug("Searching for global pool for agent with index %lu\n", params->hsa_pool_index);

    hsa_iterate_index   = 0;
    hsa_pool.handle     = (uint64_t) -1;
    status = hsa_amd_agent_iterate_memory_pools(hsa_agent, memory_pool_callback,
                                                &params->hsa_pool_index);

    if ((status != HSA_STATUS_SUCCESS) && (status != HSA_STATUS_INFO_BREAK)) {
        ucs_error("Failure to iterate regions: 0x%x\n", status);
        return UCS_ERR_NO_RESOURCE;
    }

    if (hsa_pool.handle == (uint64_t)-1) {
        ucs_error("Could not find memory pool with given index\n");
        return UCS_ERR_NO_RESOURCE;
    }

    hsa_pool.handle = hsa_pool.handle;

    return UCS_OK;
}
void rocm_shutdown()
{
    hsa_status_t status;
    ucs_debug("Shutdown HSA.....\n");

    status = hsa_shut_down();

    if (status != HSA_STATUS_SUCCESS) {
        ucs_error("Failure to close HSA connection: 0x%x\n", status);
    }
}

void *rocm_allocate_transfer_buffer(ucx_perf_params_t *params, size_t buffer_size)
{
    void *p = NULL;
    hsa_status_t status;

    status = hsa_amd_memory_pool_allocate(hsa_pool, buffer_size, 0, &p);

    if (status != HSA_STATUS_SUCCESS) {
        ucs_error("Failure to allocate HSA memory for buffer: 0x%x\n", status);
        return NULL;
    }

    status = hsa_amd_agents_allow_access(num_of_gpu,
                                         gpu_agents,
                                         NULL,  p);

    ucs_debug("Allocated HSA buffer at %p\n", p);

    if (status != HSA_STATUS_SUCCESS) {
        ucs_error("Failure to allow access for all GPUs agent. Status 0x%x\n",
                  status);

        rocm_free_transfer_buffer(p);
        p = NULL;
    }

    return p;
}

void rocm_free_transfer_buffer(void *p)
{
    hsa_status_t status;

    if (p) {
        status = hsa_amd_memory_pool_free(p);

        if (status != HSA_STATUS_SUCCESS)
            ucs_error("Failure to free HSA memory: 0x%x\n", status);
    }
}

#endif /* HAVE_ROCM */


