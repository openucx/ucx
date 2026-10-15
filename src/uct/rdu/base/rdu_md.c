/**
 * Copyright (C) SambaNova Systems, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "rdu_md.h"
#include "uct_rdu_devmem.h"
#include <uct/base/uct_md.h>
#include <uct/api/v2/uct_v2.h>
#include <uct/base/uct_iface.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucs/memory/memtype_cache.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>

/* Bounded length for the sysfs PCIe path of an RDU */
#define UCT_RDU_PATH_MAX 1024


typedef struct {
    const void *address; /* Address to look up */
    const void *base; /* Start of the allocation containing the address */
    size_t     length; /* Length of the allocation containing the address */
} uct_rdu_find_alloc_ctx_t;


static ucs_config_field_t uct_rdu_md_config_table[] =
        {{"", "", NULL, ucs_offsetof(uct_rdu_md_config_t, super),
          UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},
         {NULL}};

/*
 * Parse the BDF of a sysfs PCIe path, e.g. /sys/bus/pci/devices/0000:23:00.0,
 * into a bus id.
 */
static ucs_status_t
uct_rdu_parse_bdf(const char *path, ucs_sys_bus_id_t *bus_id_p)
{
    const char *bdf = strrchr(path, '/');
    unsigned domain, bus, slot, function;

    if (bdf == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }

    if (sscanf(bdf + 1, "%x:%x:%x.%x", &domain, &bus, &slot, &function) != 4) {
        return UCS_ERR_INVALID_PARAM;
    }

    bus_id_p->domain   = (uint16_t)domain;
    bus_id_p->bus      = (uint8_t)bus;
    bus_id_p->slot     = (uint8_t)slot;
    bus_id_p->function = (uint8_t)function;
    return UCS_OK;
}

/* PCI vendor id of SambaNova Systems */
#define UCT_RDU_PCI_VENDOR_ID 0x1e0d

/*
 * Check that a sysfs PCIe path reported by the runtime is an RDU.  Runtimes may
 * report the root complex (e.g. /sys/bus/pci/devices/0000:00:00.0) for an RDU
 * which they have not opened yet.
 */
static int uct_rdu_md_is_rdu_device(const char *path)
{
    long vendor;

    return (ucs_read_file_number(&vendor, 1, "%s/vendor", path) == UCS_OK) &&
           (vendor == UCT_RDU_PCI_VENDOR_ID);
}

/*
 * Return the system device of an RDU, resolving it from the PCIe path reported
 * by the runtime on first use.  An RDU which the runtime has not initialized
 * yet, or for which it reports a device that is not an RDU, stays unresolved,
 * and is retried on the next call.
 */
static ucs_sys_device_t
uct_rdu_md_get_sys_dev(uct_rdu_md_t *md, uint32_t rdu_id)
{
    char path[UCT_RDU_PATH_MAX];
    ucs_sys_bus_id_t bus_id;
    ucs_sys_device_t sys_dev;

    if (rdu_id >= UCT_RDU_MAX_RDUS) {
        return UCS_SYS_DEVICE_ID_UNKNOWN;
    }

    sys_dev = md->sys_dev[rdu_id];
    if (sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN) {
        return sys_dev;
    }

    if ((uct_rdu_devmem_get_pcie_path(rdu_id, path, sizeof(path)) !=
         UCT_RDU_DEVMEM_SUCCESS) ||
        !uct_rdu_md_is_rdu_device(path) ||
        (uct_rdu_parse_bdf(path, &bus_id) != UCS_OK) ||
        (ucs_topo_find_device_by_bus_id(&bus_id, &sys_dev) != UCS_OK)) {
        ucs_debug("rdu_md: could not resolve the system device of rdu %u",
                  rdu_id);
        return UCS_SYS_DEVICE_ID_UNKNOWN;
    }

    ucs_debug("rdu_md: rdu %u is %s, sys_dev %u", rdu_id, path, sys_dev);
    md->sys_dev[rdu_id] = sys_dev;
    return sys_dev;
}

static void
uct_rdu_find_alloc_cb(const void *va, size_t size, uint32_t rdu_id, void *arg)
{
    uct_rdu_find_alloc_ctx_t *ctx = arg;

    if ((ctx->address >= va) &&
        (ctx->address < UCS_PTR_BYTE_OFFSET(va, size))) {
        ctx->base   = va;
        ctx->length = size;
    }
}

static void uct_rdu_md_close(uct_md_h uct_md)
{
    ucs_free(uct_md);
}

static ucs_status_t uct_rdu_md_query(uct_md_h uct_md, uct_md_attr_v2_t *md_attr)
{
    uct_md_base_md_query(md_attr);
    md_attr->flags            = 0;
    md_attr->detect_mem_types = UCS_BIT(UCS_MEMORY_TYPE_RDU);
    /* The IB memory domain registers RDU memory with the dma-buf file
     * descriptor returned by mem_query, when the driver can export one */
    md_attr->dmabuf_mem_types = uct_rdu_devmem_dmabuf_supported() ?
                                        UCS_BIT(UCS_MEMORY_TYPE_RDU) :
                                        0;
    md_attr->access_mem_types = 0;
    md_attr->max_alloc        = 0;
    md_attr->max_reg          = ULONG_MAX;
    return UCS_OK;
}

static ucs_status_t
uct_rdu_md_detect_memory_type(uct_md_h md, const void *address, size_t length,
                              ucs_memory_type_t *mem_type_p)
{
    uint32_t rdu_id;
    uint64_t dev_addr;

    if ((address == NULL) ||
        (uct_rdu_devmem_detect(address, &rdu_id, &dev_addr) !=
         UCT_RDU_DEVMEM_SUCCESS)) {
        return UCS_ERR_INVALID_ADDR;
    }

    *mem_type_p = UCS_MEMORY_TYPE_RDU;
    return UCS_OK;
}

/*
 * Mark the memory type cache entry of an RDU allocation as RDU memory, only if
 * it is cached as unknown memory, as the UCM hooks of rdu_malloc() add it, or
 * as RDU memory by an earlier query. Memory which the cache does not know,
 * such as memory mapped by the runtime, is not added: no release event would
 * remove the entry, and later memory at that address would look like RDU
 * memory.
 */
static void
uct_rdu_md_memtype_cache_update(const void *address, size_t length,
                                const uct_rdu_find_alloc_ctx_t *alloc,
                                ucs_sys_device_t sys_dev)
{
    ucs_memory_info_t mem_info;
    uintptr_t start, end;

    /* A partial overlap is returned with a NULL base address */
    if ((ucs_memtype_cache_lookup(address, length, &mem_info) != UCS_OK) ||
        (mem_info.base_address == NULL) ||
        ((mem_info.type != UCS_MEMORY_TYPE_UNKNOWN) &&
         (mem_info.type != UCS_MEMORY_TYPE_RDU))) {
        return;
    }

    /* The cache may merge neighbouring regions, so update only where the
     * cached region and the allocation overlap */
    start = ucs_max((uintptr_t)mem_info.base_address, (uintptr_t)alloc->base);
    end   = ucs_min((uintptr_t)UCS_PTR_BYTE_OFFSET(mem_info.base_address,
                                                   mem_info.alloc_length),
                    (uintptr_t)UCS_PTR_BYTE_OFFSET(alloc->base, alloc->length));
    if (end > start) {
        ucs_memtype_cache_update((void*)start, end - start, UCS_MEMORY_TYPE_RDU,
                                 sys_dev, UCS_MEM_FLAG_REGISTRABLE);
    }
}

static ucs_status_t uct_rdu_mem_query(uct_md_h uct_md, const void *address,
                                      size_t length,
                                      uct_md_mem_attr_v2_t *mem_attr)
{
    uct_rdu_md_t *md               = ucs_derived_of(uct_md, uct_rdu_md_t);
    uct_rdu_find_alloc_ctx_t alloc = {address, address, length};
    uint64_t dmabuf_offset         = 0;
    int dmabuf_fd                  = UCT_DMABUF_FD_INVALID;
    uct_rdu_devmem_status_t status;
    uint32_t rdu_id;
    uint64_t dev_addr;
    ucs_sys_device_t sys_dev;

    if (uct_rdu_devmem_detect(address, &rdu_id, &dev_addr) !=
        UCT_RDU_DEVMEM_SUCCESS) {
        return UCS_ERR_INVALID_ADDR;
    }

    sys_dev = uct_rdu_md_get_sys_dev(md, rdu_id);

    /* Report the whole allocation, if the runtime knows it */
    uct_rdu_devmem_foreach_alloc(uct_rdu_find_alloc_cb, &alloc);

    uct_rdu_md_memtype_cache_update(address, length, &alloc, sys_dev);

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_MEM_TYPE) {
        mem_attr->mem_type = UCS_MEMORY_TYPE_RDU;
    }

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_SYS_DEV) {
        mem_attr->sys_dev = sys_dev;
    }

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_BASE_ADDRESS) {
        mem_attr->base_address = (void*)alloc.base;
    }

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_ALLOC_LENGTH) {
        mem_attr->alloc_length = alloc.length;
    }

    if (mem_attr->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_MEM_FLAGS) {
        mem_attr->mem_flags = UCS_MEM_FLAG_REGISTRABLE;
    }

    if (mem_attr->field_mask & (UCT_MD_MEM_ATTR_V2_FIELD_DMABUF_FD |
                                UCT_MD_MEM_ATTR_V2_FIELD_DMABUF_OFFSET)) {
        /* The returned offset is the position of the address within the
         * exported dma-buf; the caller owns the file descriptor */
        status = uct_rdu_devmem_dmabuf_supported() ?
                         uct_rdu_devmem_dmabuf_export(address, length,
                                                      &dmabuf_fd,
                                                      &dmabuf_offset) :
                         UCT_RDU_DEVMEM_ENOTSUPPORTED;
        if (status != UCT_RDU_DEVMEM_SUCCESS) {
            ucs_debug("rdu_md: failed to export dma-buf for %p length %zu: %d",
                      address, length, status);
            dmabuf_fd     = UCT_DMABUF_FD_INVALID;
            dmabuf_offset = 0;
        }

        if (mem_attr->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_DMABUF_FD) {
            mem_attr->dmabuf_fd = dmabuf_fd;
        } else if (dmabuf_fd != UCT_DMABUF_FD_INVALID) {
            close(dmabuf_fd);
        }

        if (mem_attr->field_mask & UCT_MD_MEM_ATTR_V2_FIELD_DMABUF_OFFSET) {
            mem_attr->dmabuf_offset = dmabuf_offset;
        }
    }

    return UCS_OK;
}

static uct_md_ops_t uct_rdu_md_ops = {
    .close     = uct_rdu_md_close,
    .query     = uct_rdu_md_query,
    .mem_alloc = (uct_md_mem_alloc_func_t)ucs_empty_function_return_unsupported,
    .mem_free  = (uct_md_mem_free_func_t)ucs_empty_function_return_unsupported,
    .mem_advise = (uct_md_mem_advise_func_t)
            ucs_empty_function_return_unsupported,
    .mem_reg    = (uct_md_mem_reg_func_t)ucs_empty_function_return_unsupported,
    .mem_dereg = (uct_md_mem_dereg_func_t)ucs_empty_function_return_unsupported,
    .mem_query  = uct_rdu_mem_query,
    .mkey_pack  = (uct_md_mkey_pack_func_t)ucs_empty_function_return_success,
    .mem_attach = (uct_md_mem_attach_func_t)
            ucs_empty_function_return_unsupported,
    .mem_elem_pack      = (uct_md_mem_elem_pack_func_t)
            ucs_empty_function_return_unsupported,
    .detect_memory_type = uct_rdu_md_detect_memory_type,
};

static ucs_status_t
uct_rdu_md_open(uct_component_t *component, const char *md_name,
                const uct_md_config_t *md_config, uct_md_h *md_p)
{
    uct_rdu_md_t *md;
    uint32_t rdu_id;

    /* Without the runtime there is no RDU memory to detect */
    if (uct_rdu_devmem_get() == NULL) {
        ucs_debug("rdu_md: rdu_devmem runtime is not available");
        return UCS_ERR_NO_DEVICE;
    }

    md = ucs_calloc(1, sizeof(*md), "uct_rdu_md_t");
    if (md == NULL) {
        ucs_error("rdu_md: failed to allocate uct_rdu_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops       = &uct_rdu_md_ops;
    md->super.component = &uct_rdu_component;

    /* Resolve the RDUs already initialized by the runtime; others are resolved
     * by mem_query on first use */
    for (rdu_id = 0; rdu_id < UCT_RDU_MAX_RDUS; rdu_id++) {
        md->sys_dev[rdu_id] = UCS_SYS_DEVICE_ID_UNKNOWN;
        uct_rdu_md_get_sys_dev(md, rdu_id);
    }

    *md_p = (uct_md_h)md;
    return UCS_OK;
}

/*
 * The transport has no data-movement operations and no endpoints.  It exists
 * only so that UCP keeps the memory domain, which detects RDU memory.
 */

typedef struct uct_rdu_iface {
    uct_base_iface_t super;
} uct_rdu_iface_t;

static ucs_status_t
uct_rdu_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_base_iface_query(ucs_derived_of(tl_iface, uct_base_iface_t),
                         iface_attr);
    iface_attr->cap.flags = UCT_IFACE_FLAG_CONNECT_TO_IFACE;
    iface_attr->bandwidth.dedicated =
            0.0001; /* DBL_MIN rounds to 0 when packing */
    iface_attr->bandwidth.shared = 0;
    iface_attr->max_num_eps      = 0;
    return UCS_OK;
}

static uct_iface_internal_ops_t uct_rdu_iface_internal_ops = {
    .iface_query_v2      = uct_iface_base_query_v2,
    .iface_estimate_perf = (uct_iface_estimate_perf_func_t)
            ucs_empty_function_return_unsupported,
    .iface_vfs_refresh   = (uct_iface_vfs_refresh_func_t)ucs_empty_function,
    .ep_query = (uct_ep_query_func_t)ucs_empty_function_return_unsupported,
    .ep_invalidate         = (uct_ep_invalidate_func_t)
            ucs_empty_function_return_unsupported,
    .ep_outstanding_purge  = (uct_ep_outstanding_purge_func_t)
            ucs_empty_function_return_unsupported,
    .ep_connect_to_ep_v2   = (uct_ep_connect_to_ep_v2_func_t)
            ucs_empty_function_return_unsupported,
    .iface_is_reachable_v2 = (uct_iface_is_reachable_v2_func_t)
            ucs_empty_function_return_zero,
    .ep_is_connected       = (uct_ep_is_connected_func_t)
            ucs_empty_function_return_zero_int,
};

static uct_iface_ops_t uct_rdu_iface_ops = {
    .ep_pending_purge = (uct_ep_pending_purge_func_t)ucs_empty_function,
    .ep_connect    = (uct_ep_connect_func_t)ucs_empty_function_return_success,
    .ep_disconnect = (uct_ep_disconnect_func_t)ucs_empty_function_return_success,
    .cm_ep_conn_notify = (uct_cm_ep_conn_notify_func_t)
            ucs_empty_function_return_unsupported,
    .ep_destroy = (uct_ep_destroy_func_t)ucs_empty_function_return_unsupported,
    .ep_put_short = (uct_ep_put_short_func_t)
            ucs_empty_function_return_unsupported,
    .ep_put_bcopy = (uct_ep_put_bcopy_func_t)
            ucs_empty_function_return_unsupported,
    .ep_get_bcopy = (uct_ep_get_bcopy_func_t)
            ucs_empty_function_return_unsupported,
    .ep_am_short = (uct_ep_am_short_func_t)ucs_empty_function_return_unsupported,
    .ep_am_short_iov = (uct_ep_am_short_iov_func_t)
            ucs_empty_function_return_unsupported,
    .ep_am_bcopy = (uct_ep_am_bcopy_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic_cswap64 = (uct_ep_atomic_cswap64_func_t)
            ucs_empty_function_return_unsupported,
    .ep_atomic64_post  = (uct_ep_atomic64_post_func_t)
            ucs_empty_function_return_unsupported,
    .ep_atomic64_fetch = (uct_ep_atomic64_fetch_func_t)
            ucs_empty_function_return_unsupported,
    .ep_atomic_cswap32 = (uct_ep_atomic_cswap32_func_t)
            ucs_empty_function_return_unsupported,
    .ep_atomic32_post  = (uct_ep_atomic32_post_func_t)
            ucs_empty_function_return_unsupported,
    .ep_atomic32_fetch = (uct_ep_atomic32_fetch_func_t)
            ucs_empty_function_return_unsupported,
    .ep_pending_add    = (uct_ep_pending_add_func_t)
            ucs_empty_function_return_unsupported,
    .ep_flush    = (uct_ep_flush_func_t)ucs_empty_function_return_success,
    .ep_fence    = (uct_ep_fence_func_t)ucs_empty_function_return_unsupported,
    .ep_check    = (uct_ep_check_func_t)ucs_empty_function_return_unsupported,
    .ep_create   = (uct_ep_create_func_t)ucs_empty_function_return_unsupported,
    .iface_flush = uct_base_iface_flush,
    .iface_fence = uct_base_iface_fence,
    .iface_progress_enable    = (uct_iface_progress_enable_func_t)
            ucs_empty_function,
    .iface_progress_disable   = (uct_iface_progress_disable_func_t)
            ucs_empty_function,
    .iface_progress           = (uct_iface_progress_func_t)
            ucs_empty_function_return_zero_int,
    .iface_event_fd_get       = (uct_iface_event_fd_get_func_t)
            ucs_empty_function_return_unsupported,
    .iface_event_arm          = (uct_iface_event_arm_func_t)
            ucs_empty_function_return_unsupported,
    .iface_close              = (uct_iface_close_func_t)ucs_empty_function,
    .iface_query              = uct_rdu_iface_query,
    .iface_get_device_address = (uct_iface_get_device_address_func_t)
            ucs_empty_function_return_success,
    .iface_get_address        = (uct_iface_get_address_func_t)
            ucs_empty_function_return_success,
    .iface_is_reachable       = uct_base_iface_is_reachable,
    .ep_get_zcopy             = (uct_ep_get_zcopy_func_t)
            ucs_empty_function_return_unsupported,
    .ep_put_zcopy             = (uct_ep_put_zcopy_func_t)
            ucs_empty_function_return_unsupported,
    .ep_am_zcopy = (uct_ep_am_zcopy_func_t)ucs_empty_function_return_unsupported,
    .ep_tag_eager_short    = (uct_ep_tag_eager_short_func_t)
            ucs_empty_function_return_unsupported,
    .ep_tag_eager_bcopy    = (uct_ep_tag_eager_bcopy_func_t)
            ucs_empty_function_return_unsupported,
    .ep_tag_eager_zcopy    = (uct_ep_tag_eager_zcopy_func_t)
            ucs_empty_function_return_unsupported,
    .ep_tag_rndv_zcopy     = (uct_ep_tag_rndv_zcopy_func_t)
            ucs_empty_function_return_unsupported,
    .ep_tag_rndv_cancel    = (uct_ep_tag_rndv_cancel_func_t)
            ucs_empty_function_return_unsupported,
    .ep_tag_rndv_request   = (uct_ep_tag_rndv_request_func_t)
            ucs_empty_function_return_unsupported,
    .iface_tag_recv_zcopy  = (uct_iface_tag_recv_zcopy_func_t)
            ucs_empty_function_return_unsupported,
    .iface_tag_recv_cancel = (uct_iface_tag_recv_cancel_func_t)
            ucs_empty_function_return_unsupported,
};

UCS_CLASS_INIT_FUNC(uct_rdu_iface_t, uct_md_h md, uct_worker_h worker,
                    const uct_iface_params_t *params,
                    const uct_iface_config_t *tl_config)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_rdu_iface_ops,
                              &uct_rdu_iface_internal_ops, md, worker, params,
                              tl_config UCS_STATS_ARG(params->stats_root)
                                      UCS_STATS_ARG(UCT_RDU_TL_NAME));
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rdu_iface_t)
{
}

UCS_CLASS_DEFINE(uct_rdu_iface_t, uct_base_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rdu_iface_t, uct_iface_t, uct_md_h, uct_worker_h,
                          const uct_iface_params_t*, const uct_iface_config_t*);

static ucs_status_t
uct_rdu_query_devices(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                      unsigned *num_tl_devices_p)
{
    return uct_single_device_resource(md, "rdu", UCT_DEVICE_TYPE_ACC,
                                      UCS_SYS_DEVICE_ID_UNKNOWN, tl_devices_p,
                                      num_tl_devices_p);
}

UCT_TL_DEFINE(&uct_rdu_component, rdu, uct_rdu_query_devices, uct_rdu_iface_t,
              "RDU_", uct_iface_config_table, uct_iface_config_t);

uct_component_t uct_rdu_component = {
    .query_md_resources = uct_md_query_single_md_resource,
    .md_open            = uct_rdu_md_open,
    .cm_open            = (uct_component_cm_open_func_t)
            ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_md_stub_rkey_unpack,
    .rkey_ptr           = (uct_component_rkey_ptr_func_t)
            ucs_empty_function_return_unsupported,
    .rkey_release       = (uct_component_rkey_release_func_t)
            ucs_empty_function_return_success,
    .rkey_compare       = uct_base_rkey_compare,
    .name               = "rdu",
    .md_config =
            {
                    .name   = "RDU memory domain",
                    .prefix = "RDU_",
                    .table  = uct_rdu_md_config_table,
                    .size   = sizeof(uct_rdu_md_config_t),
            },
    .cm_config   = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .tl_list     = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_rdu_component),
    .flags       = 0,
    .md_vfs_init = (uct_component_md_vfs_init_func_t)ucs_empty_function,
};
UCT_COMPONENT_REGISTER(&uct_rdu_component);
