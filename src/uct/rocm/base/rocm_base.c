/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019-2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rocm_base.h"
#include "rocm_signal.h"

#include <ucs/sys/string.h>
#include <ucs/sys/module.h>
#include <ucs/sys/sock.h>
#include <ucs/memory/memtype_cache.h>
#include <ucs/type/init_once.h>
#include <ucs/sys/math.h>
#include <ucs/sys/sys.h>
#include <ucm/api/ucm.h>
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
            ucs_topo_sys_device_set_class(sys_dev, UCS_TOPO_DEVICE_CLASS_ACC);
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

#ifdef HAVE_ROCM_VMM_TYPE
/*
 * VMM mapping registry.
 *
 * hsa_amd_pointer_info() reports agentBaseAddress as the queried pointer (not
 * the hsa_amd_vmem_map() base) for VMEM memory, and no HSA API resolves an
 * interior VMM pointer to its mapping base. The base is observable only at
 * vmem_map() time, so this registry records every rocm mapping reported by the
 * ucm mem-type hooks (and the existing-allocation replay) and lets the IPC and
 * mem_query paths recover the exact base for a VMM pointer. It is independent
 * of the memtype cache, so it works with UCX_MEMTYPE_CACHE=n.
 *
 * Unlike the memtype cache it never merges adjacent regions: each vmem_map() is
 * a distinct allocation whose base is the exact key required by
 * hsa_amd_vmem_retain_alloc_handle(), and merging would return a neighbouring
 * mapping's base. Pool allocations are tracked too (events do not distinguish
 * them) but are never looked up on the VMM path, so they are harmless.
 */
typedef struct uct_rocm_base_vmm_region {
    ucs_pgt_region_t super;  /**< page-aligned [start, end) for the page table */
    ucs_list_link_t  list;   /**< used to collect regions for removal */
    void             *base;  /**< exact allocation base */
    size_t           size;   /**< exact allocation size */
    uint64_t         reg_id; /**< stable id for this live mapping */
} uct_rocm_base_vmm_region_t;

/* Monotonic id assigned once per live mapping (under the registry lock). Gives
 * the rocm_ipc rkey a stable identity for the same live chunk, so concurrent
 * re-registrations hit the IPC cache instead of unmapping each other. Started
 * at a high offset so it never collides with the per-registration fallback id
 * used on a registry miss. */
static uint64_t uct_rocm_base_vmm_reg_id = (1ull << 48);

static struct {
    ucs_spinlock_t lock;
    ucs_pgtable_t  pgtable;
    int            initialized;
} uct_rocm_base_vmm_registry;

static ucs_pgt_dir_t *
uct_rocm_base_vmm_pgt_dir_alloc(const ucs_pgtable_t *pgtable)
{
    void *ptr;
    int ret;

    ret = ucs_posix_memalign(&ptr,
                             ucs_max(sizeof(void*), UCS_PGT_ENTRY_MIN_ALIGN),
                             sizeof(ucs_pgt_dir_t), "rocm_vmm_pgdir");

    return (ret == 0) ? ptr : NULL;
}

static void uct_rocm_base_vmm_pgt_dir_release(const ucs_pgtable_t *pgtable,
                                              ucs_pgt_dir_t *dir)
{
    ucs_free(dir);
}

static void uct_rocm_base_vmm_region_collect(const ucs_pgtable_t *pgtable,
                                             ucs_pgt_region_t *pgt_region,
                                             void *arg)
{
    uct_rocm_base_vmm_region_t *region =
            ucs_derived_of(pgt_region, uct_rocm_base_vmm_region_t);

    ucs_list_add_tail((ucs_list_link_t*)arg, &region->list);
}

/* registry lock must be held */
static void uct_rocm_base_vmm_registry_remove(ucs_pgt_addr_t start,
                                              ucs_pgt_addr_t end)
{
    uct_rocm_base_vmm_region_t *region, *tmp;
    UCS_LIST_HEAD(region_list);

    ucs_pgtable_search_range(&uct_rocm_base_vmm_registry.pgtable, start, end - 1,
                             uct_rocm_base_vmm_region_collect, &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        ucs_pgtable_remove(&uct_rocm_base_vmm_registry.pgtable, &region->super);
        ucs_free(region);
    }
}

/* registry lock must be held */
static void uct_rocm_base_vmm_registry_insert(void *address, size_t size)
{
    ucs_pgt_addr_t start = ucs_align_down_pow2((uintptr_t)address,
                                               UCS_PGT_ADDR_ALIGN);
    ucs_pgt_addr_t end   = ucs_align_up_pow2((uintptr_t)address + size,
                                             UCS_PGT_ADDR_ALIGN);
    uct_rocm_base_vmm_region_t *region;
    ucs_status_t status;
    int ret;

    if (start >= end) {
        return;
    }

    /* replace (never merge) any region overlapping the new range */
    uct_rocm_base_vmm_registry_remove(start, end);

    ret = ucs_posix_memalign((void**)&region,
                             ucs_max(sizeof(void*), UCS_PGT_ENTRY_MIN_ALIGN),
                             sizeof(*region), "rocm_vmm_region");
    if (ret != 0) {
        ucs_warn("failed to allocate rocm vmm region");
        return;
    }

    region->super.start = start;
    region->super.end   = end;
    region->base        = address;
    region->size        = size;
    region->reg_id      = ++uct_rocm_base_vmm_reg_id;

    status = ucs_pgtable_insert(&uct_rocm_base_vmm_registry.pgtable,
                                &region->super);
    if (status != UCS_OK) {
        ucs_warn("failed to insert rocm vmm region [0x%lx..0x%lx]: %s", start,
                 end, ucs_status_string(status));
        ucs_free(region);
    }
}

static void uct_rocm_base_vmm_event_cb(ucm_event_type_t event_type,
                                       ucm_event_t *event, void *arg)
{
    void *address = event->mem_type.address;
    size_t size   = event->mem_type.size;
    hsa_amd_pointer_type_t hsa_mem_type;
    hsa_status_t status;

    if (size == 0) {
        return;
    }

    if (event_type & UCM_EVENT_MEM_TYPE_ALLOC) {
        /* Only genuine VMEM mappings belong here; skip pool/cuda/reservation events. */
        status = uct_rocm_base_get_ptr_info(address, size, NULL, NULL,
                                            &hsa_mem_type, NULL, NULL);
        if ((status != HSA_STATUS_SUCCESS) ||
            (hsa_mem_type != HSA_EXT_POINTER_TYPE_HSA_VMEM)) {
            return;
        }

        ucs_spin_lock(&uct_rocm_base_vmm_registry.lock);
        uct_rocm_base_vmm_registry_insert(address, size);
        ucs_spin_unlock(&uct_rocm_base_vmm_registry.lock);

        /* Grant GPU access once at map time to avoid disrupting in-flight copies. */
        status = uct_rocm_base_vmm_set_gpu_access(address, size);
        if (status != HSA_STATUS_SUCCESS) {
            ucs_warn("failed to grant GPU access to vmm mapping %p size 0x%zx: "
                     "%d", address, size, status);
        }
    } else if (event_type & UCM_EVENT_MEM_TYPE_FREE) {
        /* Remove by range; harmless if the range was never a VMM mapping. */
        ucs_spin_lock(&uct_rocm_base_vmm_registry.lock);
        uct_rocm_base_vmm_registry_remove(
                ucs_align_down_pow2((uintptr_t)address, UCS_PGT_ADDR_ALIGN),
                ucs_align_up_pow2((uintptr_t)address + size,
                                  UCS_PGT_ADDR_ALIGN));
        ucs_spin_unlock(&uct_rocm_base_vmm_registry.lock);
    }
}

static void uct_rocm_base_vmm_registry_init(void)
{
    ucs_status_t status;

    if (ucs_spinlock_init(&uct_rocm_base_vmm_registry.lock, 0) != 0) {
        ucs_warn("failed to init rocm vmm registry lock");
        return;
    }

    status = ucs_pgtable_init(&uct_rocm_base_vmm_registry.pgtable,
                              uct_rocm_base_vmm_pgt_dir_alloc,
                              uct_rocm_base_vmm_pgt_dir_release);
    if (status != UCS_OK) {
        ucs_warn("failed to init rocm vmm pgtable: %s",
                 ucs_status_string(status));
        ucs_spinlock_destroy(&uct_rocm_base_vmm_registry.lock);
        return;
    }

    /* EXISTING_ALLOC replays mappings created before the handler was
     * installed, which closes most of the pre-hook gap for VMM bases. */
    status = ucm_set_event_handler(UCM_EVENT_MEM_TYPE_ALLOC |
                                           UCM_EVENT_MEM_TYPE_FREE |
                                           UCM_EVENT_FLAG_EXISTING_ALLOC,
                                   1000, uct_rocm_base_vmm_event_cb, NULL);
    if (status != UCS_OK) {
        ucs_diag("failed to set rocm vmm event handler: %s",
                 ucs_status_string(status));
        ucs_pgtable_cleanup(&uct_rocm_base_vmm_registry.pgtable);
        ucs_spinlock_destroy(&uct_rocm_base_vmm_registry.lock);
        return;
    }

    uct_rocm_base_vmm_registry.initialized = 1;
}

ucs_status_t uct_rocm_base_vmm_lookup_base(const void *ptr, void **base,
                                           size_t *size, uint64_t *reg_id)
{
    ucs_pgt_region_t *pgt_region;
    uct_rocm_base_vmm_region_t *region;
    ucs_status_t status = UCS_ERR_NO_ELEM;

    if (!uct_rocm_base_vmm_registry.initialized) {
        return UCS_ERR_NO_ELEM;
    }

    ucs_spin_lock(&uct_rocm_base_vmm_registry.lock);
    pgt_region = ucs_pgtable_lookup(&uct_rocm_base_vmm_registry.pgtable,
                                    (ucs_pgt_addr_t)(uintptr_t)ptr);
    if (pgt_region != NULL) {
        region = ucs_derived_of(pgt_region, uct_rocm_base_vmm_region_t);
        *base  = region->base;
        *size  = region->size;
        if (reg_id != NULL) {
            *reg_id = region->reg_id;
        }
        status = UCS_OK;
    }
    ucs_spin_unlock(&uct_rocm_base_vmm_registry.lock);
    return status;
}

static void uct_rocm_base_vmm_found_cb(const ucs_pgtable_t *pgtable,
                                       ucs_pgt_region_t *pgt_region, void *arg)
{
    *(int*)arg = 1;
}

int uct_rocm_base_vmm_range_overlaps(const void *addr, size_t length)
{
    ucs_pgt_addr_t start = ucs_align_down_pow2((uintptr_t)addr, UCS_PGT_ADDR_ALIGN);
    ucs_pgt_addr_t end   = ucs_align_up_pow2((uintptr_t)addr + length,
                                             UCS_PGT_ADDR_ALIGN);
    int found            = 0;

    if (!uct_rocm_base_vmm_registry.initialized || (start >= end)) {
        return 0;
    }

    ucs_spin_lock(&uct_rocm_base_vmm_registry.lock);
    ucs_pgtable_search_range(&uct_rocm_base_vmm_registry.pgtable, start, end - 1,
                             uct_rocm_base_vmm_found_cb, &found);
    ucs_spin_unlock(&uct_rocm_base_vmm_registry.lock);
    return found;
}

hsa_status_t uct_rocm_base_vmm_set_gpu_access(void *va, size_t size)
{
    hsa_amd_memory_access_desc_t *descs;
    hsa_agent_t *gpu_agents;
    int num_gpu, i;

    num_gpu = uct_rocm_base_get_gpu_agents(&gpu_agents);
    if (num_gpu == 0) {
        return HSA_STATUS_SUCCESS;
    }

    descs = ucs_alloca(num_gpu * sizeof(*descs));
    for (i = 0; i < num_gpu; i++) {
        descs[i].permissions  = HSA_ACCESS_PERMISSION_RW;
        descs[i].agent_handle = gpu_agents[i];
    }

    return hsa_amd_vmem_set_access(va, size, descs, num_gpu);
}
#endif

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

#ifdef HAVE_ROCM_VMM_TYPE
    /* Best-effort: a failure leaves VMM IPC to the align-down fallback. */
    uct_rocm_base_vmm_registry_init();
#endif

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

hsa_agent_t uct_rocm_base_get_dev_agent(int gpu_num)
{
    ucs_assertv(gpu_num < uct_rocm_base_agents.num_gpu, "gpu %d, num_gpus %d",
                gpu_num, uct_rocm_base_agents.num_gpu);
    return uct_rocm_base_agents.gpu_agents[gpu_num];
}

int uct_rocm_base_get_dev_num(hsa_agent_t agent)
{
    int i;

    for (i = 0; i < uct_rocm_base_agents.num_gpu; i++) {
        if (uct_rocm_base_agents.gpu_agents[i].handle == agent.handle) {
            return i;
        }
    }
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
        /* HSA_EXT_POINTER_TYPE_RESERVED_ADDR check is required
         * for rocm7+ for hip managed memory
         */
        if ((info.type == HSA_EXT_POINTER_TYPE_UNKNOWN)
#ifdef HAVE_ROCM_RESERVED_ADDR_TYPE
            || (info.type == HSA_EXT_POINTER_TYPE_RESERVED_ADDR)
#endif
        ) {
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
        uct_rocm_base_is_gpu_hsa_mem_type(hsa_mem_type) &&
        (dev_type == HSA_DEVICE_TYPE_GPU)) {
        uct_rocm_base_last_device_agent_used = uct_rocm_base_get_dev_num(agent);
        *mem_type_p = UCS_MEMORY_TYPE_ROCM;
    }

    return UCS_OK;
}

FILE* uct_rocm_base_load_kernel_config_file()
{
    const char *fmts[] = {
        "/boot/config-%s",
        "/usr/src/linux-%s/.config",
        "/usr/lib/modules/%s/config",
        "/usr/lib/kernel/config-%s",
        "/usr/src/linux-headers-%s/.config",
        "/lib/modules/%s/build/.config",
        "/usr/lib/ostree-boot/config-%s"
    };
    FILE* fp = NULL;
    char kernel_conf_file[128];
    struct utsname utsname;
    
    if (uname(&utsname) == -1) {
        ucs_trace("could not get kernel name");
        return NULL;
    }

    for (size_t i = 0; i < sizeof(fmts) / sizeof(fmts[0]); ++i) {
        const char *path = fmts[i];
        ucs_snprintf_safe(kernel_conf_file, sizeof(kernel_conf_file), path, utsname.release);
        fp = fopen(kernel_conf_file, "r");
        if (fp != NULL) {
            ucs_trace("reading kernel config from %s", kernel_conf_file);
            return fp;
        }
        ucs_trace("could not open kernel conf file %s error: %m", kernel_conf_file);
    }
    return fp;
}

int uct_rocm_base_file_contains_dmabuf_support(FILE* fp, const char kernel_opt1[], const char kernel_opt2[])
{
    int found_opt1 = 0;
    int found_opt2 = 0;
    char buf[256];

    while (fgets(buf, sizeof(buf), fp) != NULL) {
        if (!found_opt1 && (strstr(buf, kernel_opt1) != NULL)) {
            found_opt1 = 1;
        }
        if (!found_opt2 && (strstr(buf, kernel_opt2) != NULL)) {
            found_opt2 = 1;
        }
        if (found_opt1 && found_opt2) {
            return 1;
        }
    }
    return 0;
}

int uct_rocm_base_kernel_config_supports_dmabuf()
{
    int dmabuf_supported = 0;
    FILE* fp = NULL;
    const char kernel_opt1[] = "CONFIG_DMABUF_MOVE_NOTIFY=y";
    const char kernel_opt2[] = "CONFIG_PCI_P2PDMA=y";

    fp = uct_rocm_base_load_kernel_config_file();
    if (fp == NULL) {
        ucs_trace("no kernel conf file found");
        return 0;
    }
    dmabuf_supported = uct_rocm_base_file_contains_dmabuf_support(fp, kernel_opt1, kernel_opt2);
    fclose(fp);
    return dmabuf_supported;
} 

int uct_rocm_base_kernel_symbols_supports_dmabuf()
{
    int dmabuf_supported = 0;
    FILE* fp = NULL;
    const char kernel_sym1[] = "dma_buf_move_notify";
    const char kernel_sym2[] = "pci_p2pdma";

    fp = fopen("/proc/kallsyms", "r");
    if (fp == NULL) {
        ucs_trace("no /proc/kallsyms file found");
        return dmabuf_supported;
    }
    dmabuf_supported = uct_rocm_base_file_contains_dmabuf_support(fp, kernel_sym1, kernel_sym2);
    fclose(fp);
    return dmabuf_supported;
}

int uct_rocm_base_is_dmabuf_supported()
{
    static ucs_init_once_t init_once = UCS_INIT_ONCE_INITIALIZER;
    static int dmabuf_supported      = 0;

    UCS_INIT_ONCE(&init_once) {
#if HAVE_HSA_AMD_PORTABLE_EXPORT_DMABUF
        if (uct_rocm_base_kernel_config_supports_dmabuf()) {
            dmabuf_supported = 1;
        } else {
            ucs_trace("no kernel conf file found or no support for dmabuf "
                      "found, trying /proc/kallsyms fallback");
            dmabuf_supported = uct_rocm_base_kernel_symbols_supports_dmabuf();
        }
#endif
        ucs_debug("dmabuf is%s supported on ROCm",
                  dmabuf_supported ? "" : " not");
    }

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
            ucs_debug("failed to export dmabuf handle for addr %p / %zu: 0x%x",
                      addr, length, status);
        }

        ucs_trace("dmabuf export addr %p %lu to dmabuf fd %d offset %zu\n",
                  addr, length, fd, offset);
    }
#endif
    *dmabuf_fd     = fd;
    *dmabuf_offset = (size_t)offset;
}

ucs_status_t uct_rocm_base_mem_query(uct_md_h md, const void *addr,
                                     size_t length, int have_dmabuf,
                                     uct_md_mem_attr_v2_t *mem_attr_p)
{
    size_t dmabuf_offset       = 0;
    ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_HOST;
    ucs_sys_device_t sys_dev   = UCS_SYS_DEVICE_ID_UNKNOWN;
    int dmabuf_fd              = UCT_DMABUF_FD_INVALID;
    hsa_status_t status;
    hsa_device_type_t dev_type;
    hsa_amd_pointer_type_t hsa_mem_type;
    hsa_agent_t agent;
    ucs_status_t ucs_status;
    void *base_addr;
    size_t base_size;

    status = uct_rocm_base_get_ptr_info((void*)addr, length, &base_addr,
                                        &base_size, &hsa_mem_type, &agent,
                                        &dev_type);
    if (status != HSA_STATUS_SUCCESS) {
        return UCS_ERR_INVALID_ADDR;
    }

    if (uct_rocm_base_is_gpu_hsa_mem_type(hsa_mem_type) &&
        (dev_type == HSA_DEVICE_TYPE_GPU)) {
        mem_type = UCS_MEMORY_TYPE_ROCM;
        uct_rocm_base_last_device_agent_used = uct_rocm_base_get_dev_num(agent);
        ucs_status = uct_rocm_base_get_sys_dev(agent, &sys_dev);
        if (ucs_status != UCS_OK) {
            sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
        }
#ifdef HAVE_ROCM_VMM_TYPE
        if (hsa_mem_type == HSA_EXT_POINTER_TYPE_HSA_VMEM) {
            void *vmm_base;
            size_t vmm_size;

            /* agentBaseAddress is the queried pointer for VMEM, not the mapping
             * base; use the true base/size from the vmm registry so the
             * reported base_address and the memtype cache update below are not
             * poisoned. */
            if (uct_rocm_base_vmm_lookup_base(addr, &vmm_base, &vmm_size,
                                              NULL) == UCS_OK) {
                base_addr = vmm_base;
                base_size = vmm_size;
            }
        }
#endif
    } else {
        base_addr = (void*)addr;
        base_size = length;
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_MEM_TYPE) {
        mem_attr_p->mem_type = mem_type;
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_SYS_DEV) {
        mem_attr_p->sys_dev = sys_dev;
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_BASE_ADDRESS) {
        mem_attr_p->base_address = base_addr;
    }

    if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_ALLOC_LENGTH) {
        mem_attr_p->alloc_length = base_size;
    }

    if ((mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_DMABUF_FD) ||
        (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_DMABUF_OFFSET)) {
        if (have_dmabuf) {
            uct_rocm_base_dmabuf_export(base_addr, base_size, mem_type,
                                        &dmabuf_fd, &dmabuf_offset);
        }

        /* Offset is defined only relative to a valid dmabuf_fd. Check the fd
         * before close, which sets it to -1 on the offset-only query path. */
        if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_DMABUF_OFFSET) {
            mem_attr_p->dmabuf_offset = 0;
            if (dmabuf_fd != UCT_DMABUF_FD_INVALID) {
                mem_attr_p->dmabuf_offset = dmabuf_offset +
                                            UCS_PTR_BYTE_DIFF(base_addr, addr);
            }
        }

        if (mem_attr_p->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_DMABUF_FD) {
            mem_attr_p->dmabuf_fd = dmabuf_fd;
        } else {
            ucs_close_fd(&dmabuf_fd);
        }
    }

    if (mem_type == UCS_MEMORY_TYPE_ROCM) {
        ucs_memtype_cache_update(base_addr, base_size, mem_type, sys_dev,
                                 UCS_MEM_FLAG_REGISTRABLE);
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
    char gfx_name[64];
    hsa_status_t status;

    /* Query the gfx architecture name */
    status = hsa_agent_get_info(uct_rocm_base_agents.gpu_agents[0],
                                (hsa_agent_info_t)HSA_AGENT_INFO_NAME,
                                gfx_name);
    if (status != HSA_STATUS_SUCCESS) {
        ucs_debug("hsa_agent_get_info failed: %d", status);
        return gpu_product;
    }

    if (NULL != strstr(gfx_name, "gfx94")) {
        /* This is an MI300/325 GPU */
        gpu_product = UCT_ROCM_AMD_GPU_MI300;
        ucs_trace("found gfx94* GPU");
    } else if (NULL != strstr(gfx_name, "gfx95")) {
        /* This is an MI35x GPU */
        gpu_product = UCT_ROCM_AMD_GPU_MI350;
        ucs_trace("found gfx950 GPU");
    } else {
        ucs_trace("assuming MI100/MI200 GPU");
    }

    return gpu_product;
}

ucs_status_t uct_rocm_base_ep_flush(uct_ep_h tl_ep, ucs_mpool_t *signal_pool,
                                    ucs_queue_head_t *signal_queue,
                                    uct_completion_t *comp)
{
    uct_rocm_base_signal_desc_t *flush_signal;

    if (ucs_queue_is_empty(signal_queue)) {
        UCT_TL_EP_STAT_FLUSH(ucs_derived_of(tl_ep, uct_base_ep_t));
        return UCS_OK;
    }

    if (comp == NULL) {
        UCT_TL_EP_STAT_FLUSH_WAIT(ucs_derived_of(tl_ep, uct_base_ep_t));
        return UCS_INPROGRESS;
    }

    flush_signal = ucs_mpool_get(signal_pool);
    if (flush_signal == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    hsa_signal_store_screlease(flush_signal->signal, 0);
    flush_signal->comp        = comp;
    flush_signal->mapped_addr = NULL;
    ucs_queue_push(signal_queue, &flush_signal->queue);

    UCT_TL_EP_STAT_FLUSH_WAIT(ucs_derived_of(tl_ep, uct_base_ep_t));
    return UCS_INPROGRESS;
}

UCS_MODULE_INIT() {
    UCS_MODULE_FRAMEWORK_DECLARE(uct_rocm);
    UCS_MODULE_FRAMEWORK_LOAD(uct_rocm, 0);
    return UCS_OK;
}
