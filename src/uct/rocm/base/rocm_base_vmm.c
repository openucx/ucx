/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019-2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rocm_base.h"

#include <ucs/datastruct/pgtable.h>
#include <ucs/datastruct/list.h>
#include <ucs/type/spinlock.h>
#include <ucs/type/status.h>
#include <ucs/debug/log.h>
#include <ucs/sys/math.h>
#include <ucs/sys/sys.h>
#include <ucm/api/ucm.h>

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
 * mapping's base; adjacency also lets the IPC path flag pool chunks (is_pooled).
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
            ucs_debug("failed to grant GPU access to vmm mapping %p size 0x%zx: "
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

void uct_rocm_base_vmm_registry_init(void)
{
    ucs_status_t status;

    if (ucs_spinlock_init(&uct_rocm_base_vmm_registry.lock, 0) != UCS_OK) {
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

/* Symmetric teardown for uct_rocm_base_vmm_registry_init: unhook the event
 * handler before the module unloads, then free all regions and the pgtable. */
void uct_rocm_base_vmm_registry_cleanup(void)
{
    uct_rocm_base_vmm_region_t *region, *tmp;
    UCS_LIST_HEAD(region_list);

    if (!uct_rocm_base_vmm_registry.initialized) {
        return;
    }

    ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_ALLOC | UCM_EVENT_MEM_TYPE_FREE,
                            uct_rocm_base_vmm_event_cb, NULL);

    ucs_pgtable_purge(&uct_rocm_base_vmm_registry.pgtable,
                      uct_rocm_base_vmm_region_collect, &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        ucs_free(region);
    }

    ucs_pgtable_cleanup(&uct_rocm_base_vmm_registry.pgtable);
    ucs_spinlock_destroy(&uct_rocm_base_vmm_registry.lock);
    uct_rocm_base_vmm_registry.initialized = 0;
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

/* Non-zero if [base, base+size) abuts another VMM mapping -- the signal the IPC
 * path uses to treat it as a pool chunk (e.g. hipMallocAsync) and stage it. */
int uct_rocm_base_vmm_is_pooled(const void *base, size_t size)
{
    uintptr_t start = (uintptr_t)base;
    uintptr_t end   = start + size;
    int pooled;

    if (!uct_rocm_base_vmm_registry.initialized) {
        return 0;
    }

    ucs_spin_lock(&uct_rocm_base_vmm_registry.lock);
    pooled = (ucs_pgtable_lookup(&uct_rocm_base_vmm_registry.pgtable,
                                 (ucs_pgt_addr_t)end) != NULL) ||
             ((start > 0) &&
              (ucs_pgtable_lookup(&uct_rocm_base_vmm_registry.pgtable,
                                  (ucs_pgt_addr_t)(start - 1)) != NULL));
    ucs_spin_unlock(&uct_rocm_base_vmm_registry.lock);
    return pooled;
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
