/**
 * Copyright (C) SambaNova Systems, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <tools/perf/lib/libperf_int.h>
#include <uct/rdu/base/uct_rdu_devmem.h>

#include <ucs/sys/compiler.h>
#include <ucs/debug/log.h>
#include <ucs/memory/memtype_cache.h>
#include <ucs/sys/sock.h>

#include <pthread.h>
#include <string.h>


/*
 * RDU memory is allocated with the SambaNova rdu_devmem runtime, which the RDU
 * memory domain loads at run time. The functions are called through pointers,
 * which the UCM hooks don't intercept, so the allocator updates the memory
 * type cache as the hooks do. RDU memory is mapped to the host address space,
 * so it is copied and set with the host functions.
 */

static pthread_once_t ucx_perf_rdu_init_once = PTHREAD_ONCE_INIT;
static ucs_status_t ucx_perf_rdu_init_status = UCS_ERR_NO_DEVICE;
static const uct_rdu_devmem_ops_t *ucx_perf_rdu_ops;
static uint32_t ucx_perf_rdu_id;


static void ucx_perf_rdu_do_init(void)
{
    const uct_rdu_devmem_ops_t *ops = uct_rdu_devmem_get();
    int ret;

    if (ops == NULL) {
        ucs_error("failed to load the rdu_devmem runtime");
        return;
    }

    if ((ops->init == NULL) || (ops->malloc == NULL) || (ops->free == NULL)) {
        ucs_error("the rdu_devmem runtime is missing allocation functions");
        return;
    }

    ret = ops->init();
    if (ret != 0) {
        ucs_error("rdu_devmem_init() failed: %d", ret);
        return;
    }

    ucx_perf_rdu_ops         = ops;
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

    address = ucx_perf_rdu_ops->malloc(ucx_perf_rdu_id, length, mem_kind);
    if (address == NULL) {
        ucs_error("rdu_malloc(rdu=%u, length=%zu, kind=%d) failed",
                  ucx_perf_rdu_id, length, mem_kind);
        return UCS_ERR_NO_MEMORY;
    }

    *address_p = address;
    return UCS_OK;
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
    return ucx_perf_rdu_mem_alloc(length, UCT_RDU_DEVMEM_MEM_HBM, address_p);
}

static ucs_status_t ucx_perf_rdu_ddr_mem_alloc(const ucx_perf_context_t *perf,
                                               size_t length, void **address_p)
{
    return ucx_perf_rdu_mem_alloc(length, UCT_RDU_DEVMEM_MEM_DDR, address_p);
}

static void ucx_perf_rdu_mem_free(const ucx_perf_context_t *perf, void *address)
{
    const void *base;
    size_t length;

    /* Remove the buffer from the memory type cache, as the rdu_free() hook
     * does, since the runtime reuses the address range */
    if ((ucs_memtype_cache_global_instance != NULL) &&
        uct_rdu_devmem_find_alloc(address, &base, &length)) {
        ucs_memtype_cache_remove(base, length);
    }

    ucx_perf_rdu_ops->free(ucx_perf_rdu_id, address);
}

/*
 * Register RDU memory with the memory domain under test. When the runtime can
 * export RDU memory as a dma-buf, pass it, as UCP does with the file descriptor
 * returned by the RDU memory domain: some kernels can register RDU memory only
 * through a dma-buf.
 */
static ucs_status_t uct_perf_rdu_mem_reg(uct_md_h md, void *address,
                                         size_t length, unsigned flags,
                                         uct_mem_h *memh_p)
{
    int dmabuf_fd         = UCT_DMABUF_FD_INVALID;
    uint64_t dmabuf_offset = 0;
    uct_md_mem_reg_params_t params;
    ucs_status_t status;

    params.field_mask = UCT_MD_MEM_REG_FIELD_FLAGS |
                        UCT_MD_MEM_REG_FIELD_MEM_TYPE;
    params.flags      = flags;
    params.mem_type   = UCS_MEMORY_TYPE_RDU;

    if (uct_rdu_devmem_dmabuf_supported() &&
        (uct_rdu_devmem_dmabuf_export(address, length, &dmabuf_fd,
                                      &dmabuf_offset) ==
         UCT_RDU_DEVMEM_SUCCESS)) {
        params.field_mask   |= UCT_MD_MEM_REG_FIELD_DMABUF_FD |
                               UCT_MD_MEM_REG_FIELD_DMABUF_OFFSET;
        params.dmabuf_fd     = dmabuf_fd;
        params.dmabuf_offset = dmabuf_offset;
    }

    status = uct_md_mem_reg_v2(md, address, length, &params, memh_p);
    if (dmabuf_fd != UCT_DMABUF_FD_INVALID) {
        ucs_close_fd(&dmabuf_fd);
    }

    return status;
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

    status = uct_perf_rdu_mem_reg(perf->uct.md, alloc_mem->address, length,
                                  flags, &alloc_mem->memh);
    if (status != UCS_OK) {
        ucs_error("failed to register memory");
        ucx_perf_rdu_ops->free(ucx_perf_rdu_id, alloc_mem->address);
        return status;
    }

    alloc_mem->mem_type = UCS_MEMORY_TYPE_RDU;
    alloc_mem->md       = perf->uct.md;
    alloc_mem->length   = length;

    return UCS_OK;
}

static ucs_status_t uct_perf_rdu_hbm_alloc(const ucx_perf_context_t *perf,
                                           size_t length, unsigned flags,
                                           uct_allocated_memory_t *alloc_mem)
{
    return uct_perf_rdu_alloc_reg_mem(perf, length, UCT_RDU_DEVMEM_MEM_HBM, flags,
                                      alloc_mem);
}

static ucs_status_t uct_perf_rdu_ddr_alloc(const ucx_perf_context_t *perf,
                                           size_t length, unsigned flags,
                                           uct_allocated_memory_t *alloc_mem)
{
    return uct_perf_rdu_alloc_reg_mem(perf, length, UCT_RDU_DEVMEM_MEM_DDR, flags,
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

    ucx_perf_rdu_ops->free(ucx_perf_rdu_id, alloc_mem->address);
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
