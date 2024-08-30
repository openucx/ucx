/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019-2023. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rocm_base.h"

#include <ucs/sys/string.h>
#include <ucs/sys/module.h>
#include <sys/utsname.h>
#include <pthread.h>

#define MAX_AGENTS 127
static struct agents {
    int num;
    hsa_agent_t agents[MAX_AGENTS];
    int num_gpu;
    hsa_agent_t gpu_agents[MAX_AGENTS];
} uct_rocm_base_agents;

static int uct_rocm_base_last_device_agent_used = -1;

int uct_rocm_base_get_gpu_agents(hsa_agent_t **agents)
{
    *agents = uct_rocm_base_agents.gpu_agents;
    return uct_rocm_base_agents.num_gpu;
}

static ucs_status_t
uct_rocm_base_get_sys_dev(hsa_agent_t agent, ucs_sys_device_t *sys_dev_p)
{
    hsa_status_t status;
    ucs_sys_bus_id_t bus_id;
    uint32_t bdfid;
    uint32_t domainid;

    status = hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID,
                                &bdfid);
    if (status != HSA_STATUS_SUCCESS) {
        return UCS_ERR_UNSUPPORTED;
    }
    bus_id.bus  = (bdfid & (0xFF << 8)) >> 8;
    bus_id.slot = (bdfid & (0x1F << 3)) >> 3;

    status = hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN,
                                &domainid);
    if (status != HSA_STATUS_SUCCESS) {
        return UCS_ERR_UNSUPPORTED;
    }
    bus_id.domain = domainid;

    /* function is always set to 0 */
    bus_id.function = 0;

    return ucs_topo_find_device_by_bus_id(&bus_id, sys_dev_p);
}

static void
uct_rocm_base_get_initial_device(ucs_sys_device_t *sys_dev_p)
{
    ucs_sys_device_t sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
    int last_agent = uct_rocm_base_last_device_agent_used;
    char* env_value;
    int device;

    if (uct_rocm_base_agents.num_gpu == 1) {
        uct_rocm_base_get_sys_dev(uct_rocm_base_agents.gpu_agents[0],
                                  &sys_dev);
    } else if (uct_rocm_base_last_device_agent_used != -1) {
        /* there was already a memory pointer identified as ROCm memory
           which allowed us to identify the GPU that is being used by
           this process */
        uct_rocm_base_get_sys_dev(uct_rocm_base_agents.agents[last_agent],
                                  &sys_dev);
    } else {
        /* check HIP_VISIBLE_DEVICES. Only use it if the environment variable
           is restricting the view to a single device. */
        env_value = getenv("HIP_VISIBLE_DEVICES");
        if ((env_value != NULL) && (!strchr(env_value, ','))) {
            device = atoi(env_value);
            uct_rocm_base_get_sys_dev(uct_rocm_base_agents.gpu_agents[device],
                                      &sys_dev);
        }
    }

    *sys_dev_p = sys_dev;
}

static hsa_status_t uct_rocm_hsa_agent_callback(hsa_agent_t agent, void* data)
{
    const unsigned sys_device_priority = 10;
    hsa_device_type_t device_type;
    ucs_sys_device_t sys_dev;
    char device_name[10];
    ucs_status_t status;

    ucs_assert(uct_rocm_base_agents.num < MAX_AGENTS);

    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
    if (device_type == HSA_DEVICE_TYPE_CPU) {
        ucs_trace("found cpu agent %lu", agent.handle);
    }
    else if (device_type == HSA_DEVICE_TYPE_GPU) {
        uct_rocm_base_agents.gpu_agents[uct_rocm_base_agents.num_gpu] = agent;

        status = uct_rocm_base_get_sys_dev(agent, &sys_dev);
        if (status == UCS_OK) {
            ucs_snprintf_safe(device_name, sizeof(device_name), "GPU%d",
                              uct_rocm_base_agents.num_gpu);
            ucs_topo_sys_device_set_name(sys_dev, device_name,
                                         sys_device_priority);
        }
        ucs_trace("found gpu agent %lu", agent.handle);
        uct_rocm_base_agents.num_gpu++;
    }
    else {
        ucs_trace("found unknown agent %lu", agent.handle);
    }

    uct_rocm_base_agents.agents[uct_rocm_base_agents.num] = agent;
    uct_rocm_base_agents.num++;

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
    ucs_sys_device_t sys_dev;

    uct_rocm_base_get_initial_device(&sys_dev);
    return uct_single_device_resource(md, md->component->name,
                                      UCT_DEVICE_TYPE_ACC,
                                      sys_dev, tl_devices_p,
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

hsa_status_t uct_rocm_base_get_ptr_info(void *ptr, size_t size, void **base_ptr,
                                        size_t *base_size,
                                        hsa_amd_pointer_type_t *hsa_mem_type,
                                        hsa_agent_t *agent,
                                        hsa_device_type_t *dev_type)
{
    hsa_status_t status;
    hsa_amd_pointer_info_t info;

    info.size = sizeof(hsa_amd_pointer_info_t);
    status = hsa_amd_pointer_info(ptr, &info, NULL, NULL, NULL);
    if (status != HSA_STATUS_SUCCESS) {
        ucs_error("get pointer info fail %p", ptr);
        return status;
    }

    if (hsa_mem_type != NULL) {
        *hsa_mem_type = info.type;
    }
    if (agent != NULL) {
        *agent = info.agentOwner;
    }
    if (base_ptr != NULL) {
        *base_ptr = info.agentBaseAddress;
    }
    if (base_size != NULL) {
        *base_size = info.sizeInBytes;
    }
    if (dev_type != NULL) {
        if (info.type == HSA_EXT_POINTER_TYPE_UNKNOWN) {
            *dev_type = HSA_DEVICE_TYPE_CPU;
        } else {
            status = hsa_agent_get_info(info.agentOwner, HSA_AGENT_INFO_DEVICE,
                                        dev_type);
        }
    }

    return status;
}

ucs_status_t uct_rocm_base_detect_memory_type(uct_md_h md, const void *addr,
                                              size_t length,
                                              ucs_memory_type_t *mem_type_p)
{
    hsa_status_t status;
    hsa_device_type_t dev_type;
    hsa_amd_pointer_type_t hsa_mem_type;
    hsa_agent_t agent;

    *mem_type_p = UCS_MEMORY_TYPE_HOST;
    if (addr == NULL) {
        return UCS_OK;
    }

    status = uct_rocm_base_get_ptr_info((void *)addr, length, NULL, NULL,
                                        &hsa_mem_type, &agent, &dev_type);
    if ((status == HSA_STATUS_SUCCESS) &&
        (hsa_mem_type == HSA_EXT_POINTER_TYPE_HSA) &&
        (dev_type == HSA_DEVICE_TYPE_GPU)) {
        uct_rocm_base_last_device_agent_used = uct_rocm_base_get_dev_num(agent);
        *mem_type_p = UCS_MEMORY_TYPE_ROCM;
    }

    return UCS_OK;
}

int uct_rocm_base_is_dmabuf_supported()
{
    int dmabuf_supported = 0;

#if HAVE_HSA_AMD_PORTABLE_EXPORT_DMABUF
    const char kernel_opt1[] = "CONFIG_DMABUF_MOVE_NOTIFY=y";
    const char kernel_opt2[] = "CONFIG_PCI_P2PDMA=y";
    int found_opt1           = 0;
    int found_opt2           = 0;
    FILE *fp;
    struct utsname utsname;
    char kernel_conf_file[128];
    char buf[256];

    if (uname(&utsname) == -1) {
        ucs_trace("could not get kernel name");
        goto out;
    }

    ucs_snprintf_safe(kernel_conf_file, sizeof(kernel_conf_file),
                      "/boot/config-%s", utsname.release);
    fp = fopen(kernel_conf_file, "r");
    if (fp == NULL) {
        ucs_trace("could not open kernel conf file %s error: %m",
                  kernel_conf_file);
        goto out;
    }

    while (fgets(buf, sizeof(buf), fp) != NULL) {
        if (strstr(buf, kernel_opt1) != NULL) {
            found_opt1 = 1;
        }
        if (strstr(buf, kernel_opt2) != NULL) {
            found_opt2 = 1;
        }
        if (found_opt1 && found_opt2) {
            dmabuf_supported = 1;
            break;
        }
    }
    fclose(fp);
#endif
out:
    return dmabuf_supported;
}

static void uct_rocm_base_dmabuf_export(const void *addr, const size_t length,
                                        ucs_memory_type_t mem_type,
                                        int *dmabuf_fd, size_t *dmabuf_offset)
{
    int fd          = UCT_DMABUF_FD_INVALID;
    uint64_t offset = 0;
#if HAVE_HSA_AMD_PORTABLE_EXPORT_DMABUF
    hsa_status_t status;

    if (mem_type == UCS_MEMORY_TYPE_ROCM) {
        status = hsa_amd_portable_export_dmabuf(addr, length, &fd, &offset);
        if (status != HSA_STATUS_SUCCESS) {
            fd     = UCT_DMABUF_FD_INVALID;
            offset = 0;
            ucs_warn("failed to export dmabuf handle for addr %p / %zu", addr,
                     length);
        }

        ucs_trace("dmabuf export addr %p %lu to dmabuf fd %d offset %zu\n",
                  addr, length, fd, offset);
    }
#endif
    *dmabuf_fd     = fd;
    *dmabuf_offset = (size_t)offset;
}

ucs_status_t uct_rocm_base_mem_query(uct_md_h md, const void *addr,
                                     const size_t length,
                                     uct_md_mem_attr_t *mem_attr_p)
{
    size_t dmabuf_offset       = 0;
    int is_exported            = 0;
    ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_HOST;
    int dmabuf_fd;
    hsa_status_t status;
    hsa_device_type_t dev_type;
    hsa_amd_pointer_type_t hsa_mem_type;
    hsa_agent_t agent;
    ucs_sys_device_t sys_dev;
    ucs_status_t ucs_status;

    status = uct_rocm_base_get_ptr_info((void*)addr, length, NULL, NULL,
                                        &hsa_mem_type, &agent, &dev_type);
    if (status != HSA_STATUS_SUCCESS) {
        return status;
    }

    if ((hsa_mem_type == HSA_EXT_POINTER_TYPE_HSA) &&
        (dev_type == HSA_DEVICE_TYPE_GPU)) {
        mem_type = UCS_MEMORY_TYPE_ROCM;
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_FIELD_MEM_TYPE) {
        mem_attr_p->mem_type = mem_type;
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_FIELD_SYS_DEV) {
        mem_attr_p->sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
        if (mem_type == UCS_MEMORY_TYPE_ROCM) {
            ucs_status = uct_rocm_base_get_sys_dev(agent, &sys_dev);
            if (ucs_status == UCS_OK) {
                mem_attr_p->sys_dev = sys_dev;
            }
        }
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_FIELD_BASE_ADDRESS) {
        mem_attr_p->base_address = (void*) addr;
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_FIELD_ALLOC_LENGTH) {
        mem_attr_p->alloc_length = length;
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_FIELD_DMABUF_FD) {
        uct_rocm_base_dmabuf_export(addr, length, mem_type, &dmabuf_fd,
                                    &dmabuf_offset);
        mem_attr_p->dmabuf_fd = dmabuf_fd;
        is_exported           = 1;
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_FIELD_DMABUF_OFFSET) {
        if (!is_exported) {
            uct_rocm_base_dmabuf_export(addr, length, mem_type, &dmabuf_fd,
                                        &dmabuf_offset);
        }
        mem_attr_p->dmabuf_offset = dmabuf_offset;
    }

    return UCS_OK;
}

static hsa_status_t uct_rocm_hsa_pool_callback(hsa_amd_memory_pool_t pool, void* data)
{
    int allowed;
    uint32_t flags;
    hsa_amd_segment_t segment;

    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &allowed);
    if (allowed) {
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
        if (HSA_AMD_SEGMENT_GLOBAL != segment) {
            return HSA_STATUS_SUCCESS;
        }

        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
        if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
            *((hsa_amd_memory_pool_t*)data) = pool;
            return HSA_STATUS_INFO_BREAK;
        }
    }
    return HSA_STATUS_SUCCESS;
}

ucs_status_t uct_rocm_base_get_last_device_pool(hsa_amd_memory_pool_t *pool)
{
    hsa_agent_t agent = uct_rocm_base_agents.gpu_agents[0];
    hsa_status_t hsa_status;

    if (uct_rocm_base_last_device_agent_used != -1) {
        agent = uct_rocm_base_get_dev_agent(uct_rocm_base_last_device_agent_used);
    }
    hsa_status = hsa_amd_agent_iterate_memory_pools(agent,
                                                    uct_rocm_hsa_pool_callback,
                                                    (void*)pool);
    if ((hsa_status != HSA_STATUS_SUCCESS) &&
        (hsa_status != HSA_STATUS_INFO_BREAK)) {
        ucs_debug("could not iterate HSA memory pools: 0x%x", hsa_status);
        return UCS_ERR_UNSUPPORTED;
    }

    return UCS_OK;
}

ucs_status_t uct_rocm_base_get_link_type(hsa_amd_link_info_type_t *link_type)
{
    hsa_amd_memory_pool_link_info_t link_info;
    hsa_agent_t agent1, agent2;
    hsa_amd_memory_pool_t pool;
    hsa_status_t status;

    *link_type = HSA_AMD_LINK_INFO_TYPE_PCIE;

    if (uct_rocm_base_agents.num_gpu < 2) {
        return UCS_OK;
    }

    agent1 = uct_rocm_base_agents.gpu_agents[0];
    agent2 = uct_rocm_base_agents.gpu_agents[1];

    status = hsa_amd_agent_iterate_memory_pools(agent2,
                            uct_rocm_hsa_pool_callback, (void*)&pool);
    if ((status != HSA_STATUS_SUCCESS) && (status != HSA_STATUS_INFO_BREAK)) {
        ucs_debug("Could not iterate HSA memory pools: 0x%x", status);
        return UCS_ERR_UNSUPPORTED;
    }

    status = hsa_amd_agent_memory_pool_get_info(agent1, pool,
                        HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO, &link_info);
    if (status != HSA_STATUS_SUCCESS) {
        ucs_debug("Could not get HSA memory pool info: 0x%x", status);
        return UCS_ERR_UNSUPPORTED;
    }

    *link_type = link_info.link_type;
    return UCS_OK;
}

uct_rocm_amd_gpu_product_t uct_rocm_base_get_gpu_product(void)
{
    uct_rocm_amd_gpu_product_t gpu_product = UCT_ROCM_AMD_GPU_MI200;
    char product_name[64];
    char gfx_name[64];
    hsa_status_t status;

    /* fetching data from GPU 0, assuming all GPUs on a node are
       identical */
    status = hsa_agent_get_info(uct_rocm_base_agents.gpu_agents[0],
                                (hsa_agent_info_t)
                                        HSA_AMD_AGENT_INFO_PRODUCT_NAME,
                                (void*)product_name);
    if (status != HSA_STATUS_SUCCESS) {
        ucs_debug("Error in hsa_agent_info %d", status);
        return gpu_product;
    }

    if (NULL != strstr(product_name, "MI300A")) {
        gpu_product = UCT_ROCM_AMD_GPU_MI300A;
        ucs_trace("found MI300A GPU");
    } else if (NULL != strstr(product_name, "MI300X")) {
        gpu_product = UCT_ROCM_AMD_GPU_MI300X;
        ucs_trace("found MI300X GPU");
    } else {
        /* In case product_name is not set correctly, query the gfx
           architecture name */
        status = hsa_agent_get_info(uct_rocm_base_agents.gpu_agents[0],
                                    (hsa_agent_info_t)HSA_AGENT_INFO_NAME,
                                    (void*)gfx_name);
        if (status != HSA_STATUS_SUCCESS) {
            ucs_debug("error in hsa_agent_info %d", status);
            return gpu_product;
        }

        if (NULL != strstr(gfx_name, "gfx94")) {
            /* This is an MI300 GPU, but cannot say whether its the A or X
               variant. Assuming A variant for now*/
            gpu_product = UCT_ROCM_AMD_GPU_MI300A;
            ucs_trace("found gfx94* GPU, assuming MI300A");
        } else {
            ucs_trace("assuming MI100/MI200 GPU");
        }
    }

    return gpu_product;
}

UCS_MODULE_INIT() {
    UCS_MODULE_FRAMEWORK_DECLARE(uct_rocm);
    UCS_MODULE_FRAMEWORK_LOAD(uct_rocm, 0);
    return UCS_OK;
}
