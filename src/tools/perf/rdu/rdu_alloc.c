/**
 * Copyright (C) SambaNova Systems, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <tools/perf/lib/libperf_int.h>

#include <ucs/sys/compiler.h>
#include <ucs/debug/log.h>
#include <ucs/memory/memtype_cache.h>

#include <dlfcn.h>
#include <pthread.h>
#include <string.h>


/*
 * RDU memory is allocated with the SambaNova rdu_devmem runtime, which is
 * loaded at run time, as the RDU memory domain does, so the module builds
 * without the runtime installed. The functions are called through pointers,
 * which the UCM hooks don't intercept, so the allocator updates the memory
 * type cache as the hooks do. RDU memory is mapped to the host address space,
 * so it is copied and set with the host functions.
 */

/* Memory kinds of rdu_malloc() */
#define UCX_PERF_RDU_MEM_HBM 0
#define UCX_PERF_RDU_MEM_DDR 1

typedef int (*ucx_perf_rdu_init_func_t)(void);
typedef void *(*ucx_perf_rdu_malloc_func_t)(uint32_t rdu_id, size_t size,
                                            int mem_kind);
typedef void (*ucx_perf_rdu_free_func_t)(uint32_t rdu_id, void *ptr);
typedef void (*ucx_perf_rdu_foreach_cb_t)(const void *address, size_t size,
                                          uint32_t rdu_id, void *arg);
typedef void (*ucx_perf_rdu_foreach_func_t)(ucx_perf_rdu_foreach_cb_t cb,
                                            void *arg);

typedef struct {
    const void *address;
    size_t     size;
} ucx_perf_rdu_find_alloc_t;


static pthread_once_t ucx_perf_rdu_init_once = PTHREAD_ONCE_INIT;
static ucs_status_t ucx_perf_rdu_init_status = UCS_ERR_NO_DEVICE;
static ucx_perf_rdu_malloc_func_t ucx_perf_rdu_malloc_func;
static ucx_perf_rdu_free_func_t ucx_perf_rdu_free_func;
static ucx_perf_rdu_foreach_func_t ucx_perf_rdu_foreach_func;
static uint32_t ucx_perf_rdu_id;


static void ucx_perf_rdu_do_init(void)
{
    static const char *libs[] = UCX_RDU_DEVMEM_LIBS;
    ucx_perf_rdu_init_func_t init_func;
    void *handle = NULL;
    unsigned i;
    int ret;

    for (i = 0; (i < ucs_static_array_size(libs)) && (handle == NULL); ++i) {
        handle = dlopen(libs[i], RTLD_NOW | RTLD_GLOBAL);
    }

    if (handle == NULL) {
        ucs_error("failed to load the rdu_devmem runtime: %s", dlerror());
        return;
    }

    init_func = (ucx_perf_rdu_init_func_t)dlsym(handle, "rdu_devmem_init");
    ucx_perf_rdu_malloc_func  = (ucx_perf_rdu_malloc_func_t)dlsym(handle,
                                                                  "rdu_malloc");
    ucx_perf_rdu_free_func    = (ucx_perf_rdu_free_func_t)dlsym(handle,
                                                                "rdu_free");
    ucx_perf_rdu_foreach_func = (ucx_perf_rdu_foreach_func_t)
            dlsym(handle, "rdu_foreach_alloc");
    if ((init_func == NULL) || (ucx_perf_rdu_malloc_func == NULL) ||
        (ucx_perf_rdu_free_func == NULL) ||
        (ucx_perf_rdu_foreach_func == NULL)) {
        ucs_error("the rdu_devmem runtime is missing required symbols");
        return;
    }

    ret = init_func();
    if (ret != 0) {
        ucs_error("rdu_devmem_init() failed: %d", ret);
        return;
    }

    ucx_perf_rdu_init_status = UCS_OK;
}

static ucs_status_t ucx_perf_rdu_init(ucx_perf_context_t *perf)
{
    int device_id;

    pthread_once(&ucx_perf_rdu_init_once, ucx_perf_rdu_do_init);
    if (ucx_perf_rdu_init_status != UCS_OK) {
        return ucx_perf_rdu_init_status;
    }

    /* The runtime does not report the number of RDUs, so use the requested
     * RDU, or RDU 0 by default */
    device_id = ucx_perf_get_device_id(perf, rte_call(perf, group_index));

    if (device_id == UCX_PERF_MEM_DEV_DEFAULT) {
        device_id = 0;
    } else if (device_id < 0) {
        ucs_error("rdu device index %d is invalid", device_id);
        return UCS_ERR_INVALID_PARAM;
    }

    ucx_perf_rdu_id = device_id;
    return UCS_OK;
}

static ucs_status_t
ucx_perf_rdu_alloc(size_t length, int mem_kind, void **address_p)
{
    void *address;

    address = ucx_perf_rdu_malloc_func(ucx_perf_rdu_id, length, mem_kind);
    if (address == NULL) {
        ucs_error("rdu_malloc(rdu=%u, length=%zu, kind=%d) failed",
                  ucx_perf_rdu_id, length, mem_kind);
        return UCS_ERR_NO_MEMORY;
    }

    *address_p = address;
    return UCS_OK;
}

static void ucx_perf_rdu_find_alloc_cb(const void *address, size_t size,
                                       uint32_t rdu_id, void *arg)
{
    ucx_perf_rdu_find_alloc_t *find = arg;

    if (address == find->address) {
        find->size = size;
    }
}

static ucs_status_t
ucx_perf_rdu_mem_alloc(size_t length, int mem_kind, void **address_p)
{
    ucs_status_t status;

    status = ucx_perf_rdu_alloc(length, mem_kind, address_p);
    if (status != UCS_OK) {
        return status;
    }

    /* Add the buffer as unknown memory, as the rdu_malloc() hook does, so the
     * memory type detection queries the RDU memory domain */
    ucs_memtype_cache_update(*address_p, length, UCS_MEMORY_TYPE_UNKNOWN,
                             UCS_SYS_DEVICE_ID_UNKNOWN,
                             UCS_MEM_FLAG_REGISTRABLE);
    return UCS_OK;
}

static ucs_status_t ucx_perf_rdu_hbm_mem_alloc(const ucx_perf_context_t *perf,
                                               size_t length, void **address_p)
{
    return ucx_perf_rdu_mem_alloc(length, UCX_PERF_RDU_MEM_HBM, address_p);
}

static ucs_status_t ucx_perf_rdu_ddr_mem_alloc(const ucx_perf_context_t *perf,
                                               size_t length, void **address_p)
{
    return ucx_perf_rdu_mem_alloc(length, UCX_PERF_RDU_MEM_DDR, address_p);
}

static void ucx_perf_rdu_mem_free(const ucx_perf_context_t *perf, void *address)
{
    ucx_perf_rdu_find_alloc_t find = {address, 0};

    /* Remove the buffer from the memory type cache, as the rdu_free() hook
     * does, since the runtime reuses the address range */
    if (ucs_memtype_cache_global_instance != NULL) {
        ucx_perf_rdu_foreach_func(ucx_perf_rdu_find_alloc_cb, &find);
        if (find.size != 0) {
            ucs_memtype_cache_remove(address, find.size);
        }
    }

    ucx_perf_rdu_free_func(ucx_perf_rdu_id, address);
}

static ucs_status_t
uct_perf_rdu_alloc_reg_mem(const ucx_perf_context_t *perf, size_t length,
                           int mem_kind, unsigned flags,
                           uct_allocated_memory_t *alloc_mem)
{
    ucs_status_t status;

    status = ucx_perf_rdu_alloc(length, mem_kind, &alloc_mem->address);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_perf_md_mem_reg(perf->uct.md, alloc_mem->address, length,
                                 flags, UCS_MEMORY_TYPE_RDU, &alloc_mem->memh);
    if (status != UCS_OK) {
        ucs_error("failed to register memory");
        ucx_perf_rdu_free_func(ucx_perf_rdu_id, alloc_mem->address);
        return status;
    }

    alloc_mem->mem_type = UCS_MEMORY_TYPE_RDU;
    alloc_mem->md       = perf->uct.md;

    return UCS_OK;
}

static ucs_status_t uct_perf_rdu_hbm_alloc(const ucx_perf_context_t *perf,
                                           size_t length, unsigned flags,
                                           uct_allocated_memory_t *alloc_mem)
{
    return uct_perf_rdu_alloc_reg_mem(perf, length, UCX_PERF_RDU_MEM_HBM, flags,
                                      alloc_mem);
}

static ucs_status_t uct_perf_rdu_ddr_alloc(const ucx_perf_context_t *perf,
                                           size_t length, unsigned flags,
                                           uct_allocated_memory_t *alloc_mem)
{
    return uct_perf_rdu_alloc_reg_mem(perf, length, UCX_PERF_RDU_MEM_DDR, flags,
                                      alloc_mem);
}

static void uct_perf_rdu_free(const ucx_perf_context_t *perf,
                              uct_allocated_memory_t *alloc_mem)
{
    ucs_status_t status;

    ucs_assert(alloc_mem->md == perf->uct.md);

    status = uct_md_mem_dereg(perf->uct.md, alloc_mem->memh);
    if (status != UCS_OK) {
        ucs_error("failed to deregister memory");
    }

    ucx_perf_rdu_free_func(ucx_perf_rdu_id, alloc_mem->address);
}

static void ucx_perf_rdu_memcpy(void *dst, ucs_memory_type_t dst_mem_type,
                                const void *src, ucs_memory_type_t src_mem_type,
                                size_t count)
{
    memcpy(dst, src, count);
}

static ucx_perf_allocator_t rdu_allocator = {
    .name             = "rdu",
    .flags            = UCX_PERF_ALLOCATOR_FLAG_DEVICE_ID,
    .default_mem_type = UCS_MEMORY_TYPE_RDU,
    .init             = ucx_perf_rdu_init,
    .uct_alloc        = uct_perf_rdu_hbm_alloc,
    .uct_free         = uct_perf_rdu_free,
    .mem_alloc        = ucx_perf_rdu_hbm_mem_alloc,
    .mem_free         = ucx_perf_rdu_mem_free,
    .resolve_mem_type = ucx_perf_allocator_default_resolve_mem_type,
    .memcpy           = ucx_perf_rdu_memcpy,
    .memset           = memset
};

static ucx_perf_allocator_t rdu_ddr_allocator = {
    .name             = "rdu-ddr",
    .flags            = UCX_PERF_ALLOCATOR_FLAG_DEVICE_ID,
    .default_mem_type = UCS_MEMORY_TYPE_RDU,
    .init             = ucx_perf_rdu_init,
    .uct_alloc        = uct_perf_rdu_ddr_alloc,
    .uct_free         = uct_perf_rdu_free,
    .mem_alloc        = ucx_perf_rdu_ddr_mem_alloc,
    .mem_free         = ucx_perf_rdu_mem_free,
    .resolve_mem_type = ucx_perf_allocator_default_resolve_mem_type,
    .memcpy           = ucx_perf_rdu_memcpy,
    .memset           = memset
};

UCS_STATIC_INIT
{
    ucx_perf_allocator_register(&rdu_allocator);
    ucx_perf_allocator_register(&rdu_ddr_allocator);
}

UCS_STATIC_CLEANUP
{
    ucx_perf_allocator_unregister(&rdu_ddr_allocator);
    ucx_perf_allocator_unregister(&rdu_allocator);
}
